import argparse
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Mapping

import einops
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule

from sae_lens.sae import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.load_model import load_model
import json
import os
# Everything by default is false so the user can just set the ones they want to true
@dataclass
class EvalConfig:
    # model
    model_name:str = "/root/data/sae/LLMmodel/XuanYuan-6B-Chat",
    model_class_name:str = "LlamaForCausalLM",
    device:str="cuda"
    sae_paths: list |None= None
    num_eval_batches: int = 10
    eval_batch_size_prompts: int = 8
    datasets: list |None= None
    ctx_lens: list |None= None
    save_path: str = "/root/data/sae/evals/eval_results.csv"
    n_devices: int = 3

    batch_size_prompts: int | None = None # =num_eval_batches 自动设置

    # Reconstruction metrics
    n_eval_reconstruction_batches: int = 10 # =num_eval_batches 自动设置
    compute_kl: bool = False
    compute_ce_loss: bool = False

    # Sparsity and variance metrics
    n_eval_sparsity_variance_batches: int = 1
    compute_l2_norms: bool = False
    compute_sparsity_metrics: bool = False
    compute_variance_metrics: bool = False
    def __post_init__(self):
        print(self.sae_paths)
        print(self.save_path)
        print(os.path.exists(os.path.dirname(self.save_path)))
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
        self.n_eval_reconstruction_batches=self.num_eval_batches
        self.n_eval_sparsity_variance_batches=self.num_eval_batches

        self.json_path=self.save_path+"config.json"
        self.save_path = self.save_path + "eval_results.csv"
        self.to_json(self.json_path)
    def to_dict(self) -> dict[str, Any]:

        cfg_dict = {
            **self.__dict__,
            # some args may not be serializable by default
        }
        return cfg_dict

    def to_json(self, path: str) -> None:
        # if not os.path.exists(os.path.dirname(path)):
        #     os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# def get_eval_everything_config(
#         batch_size_prompts: int | None = None,
#         n_eval_reconstruction_batches: int = 10,
#         n_eval_sparsity_variance_batches: int = 1,
# ) -> EvalConfig:
#     """
#     Returns an EvalConfig object with all metrics set to True, so that when passed to run_evals all available metrics will be run.
#     """
#     return EvalConfig(
#         batch_size_prompts=batch_size_prompts,
#         n_eval_reconstruction_batches=n_eval_reconstruction_batches,
#         compute_kl=True,
#         compute_ce_loss=True,
#         compute_l2_norms=True,
#         n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
#         compute_sparsity_metrics=True,
#         compute_variance_metrics=True,
#     )


@torch.no_grad()
def run_evals(
        sae: SAE,
        activation_store: ActivationsStore,
        model: HookedRootModule,
        eval_config: EvalConfig,
        model_kwargs: Mapping[str, Any]={} ,
) -> dict[str, Any]:
    hook_name = sae.cfg.hook_name
    actual_batch_size = (
            eval_config.batch_size_prompts or activation_store.store_batch_size_prompts
    )

    # TODO: Come up with a cleaner long term strategy here for SAEs that do reshaping.
    # turn off hook_z reshaping mode if it's on, and restore it after evals
    if "hook_z" in hook_name:
        previous_hook_z_reshaping_mode = sae.hook_z_reshaping_mode
        sae.turn_off_forward_pass_hook_z_reshaping()
    else:
        previous_hook_z_reshaping_mode = None

    metrics = {}

    if eval_config.compute_kl or eval_config.compute_ce_loss:
        assert eval_config.n_eval_reconstruction_batches > 0
        metrics |= get_downstream_reconstruction_metrics(
            sae,
            model,
            activation_store,
            compute_kl=eval_config.compute_kl,
            compute_ce_loss=eval_config.compute_ce_loss,
            n_batches=eval_config.n_eval_reconstruction_batches,
            eval_batch_size_prompts=actual_batch_size,
        )

        activation_store.reset_input_dataset()

    if (
            eval_config.compute_l2_norms
            or eval_config.compute_sparsity_metrics
            or eval_config.compute_variance_metrics
    ):
        assert eval_config.n_eval_sparsity_variance_batches > 0
        metrics |= get_sparsity_and_variance_metrics(
            sae,
            model,
            activation_store,
            compute_l2_norms=eval_config.compute_l2_norms,
            compute_sparsity_metrics=eval_config.compute_sparsity_metrics,
            compute_variance_metrics=eval_config.compute_variance_metrics,
            n_batches=eval_config.n_eval_sparsity_variance_batches,
            eval_batch_size_prompts=actual_batch_size,
            model_kwargs=model_kwargs,
        )

    if len(metrics) == 0:
        raise ValueError(
            "No metrics were computed, please set at least one metric to True."
        )

    # restore previous hook z reshaping mode if necessary
    if "hook_z" in hook_name:
        if previous_hook_z_reshaping_mode and not sae.hook_z_reshaping_mode:
            sae.turn_on_forward_pass_hook_z_reshaping()
        elif not previous_hook_z_reshaping_mode and sae.hook_z_reshaping_mode:
            sae.turn_off_forward_pass_hook_z_reshaping()

    total_tokens_evaluated = (
            activation_store.context_size
            * eval_config.n_eval_reconstruction_batches
            * actual_batch_size
    )
    metrics["metrics/total_tokens_evaluated"] = total_tokens_evaluated

    return metrics


def get_downstream_reconstruction_metrics(
        sae: SAE,
        model: HookedRootModule,
        activation_store: ActivationsStore,
        compute_kl: bool,
        compute_ce_loss: bool,
        n_batches: int,
        eval_batch_size_prompts: int,
):
    metrics_dict = {}
    if compute_kl:
        metrics_dict["kl_div_with_sae"] = []
        metrics_dict["kl_div_with_ablation"] = []
    if compute_ce_loss:
        metrics_dict["ce_loss_with_sae"] = []
        metrics_dict["ce_loss_without_sae"] = []
        metrics_dict["ce_loss_with_ablation"] = []

    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        for metric_name, metric_value in get_recons_loss(
                sae,
                model,
                batch_tokens,
                activation_store,
                compute_kl=compute_kl,
                compute_ce_loss=compute_ce_loss,
        ).items():
            metrics_dict[metric_name].append(metric_value)

    metrics: dict[str, float] = {}
    for metric_name, metric_values in metrics_dict.items():
        metrics[f"metrics/{metric_name}"] = torch.stack(metric_values).mean().item()

    if compute_kl:
        metrics["metrics/kl_div_score"] = (
                                                  metrics["metrics/kl_div_with_ablation"] - metrics[
                                              "metrics/kl_div_with_sae"]
                                          ) / metrics["metrics/kl_div_with_ablation"]

    if compute_ce_loss:
        metrics["metrics/ce_loss_score"] = (
                                                   metrics["metrics/ce_loss_with_ablation"]
                                                   - metrics["metrics/ce_loss_with_sae"]
                                           ) / (
                                                   metrics["metrics/ce_loss_with_ablation"]
                                                   - metrics["metrics/ce_loss_without_sae"]
                                           )

    return metrics


def get_sparsity_and_variance_metrics(
        sae: SAE,
        model: HookedRootModule,
        activation_store: ActivationsStore,
        n_batches: int,
        compute_l2_norms: bool,
        compute_sparsity_metrics: bool,
        compute_variance_metrics: bool,
        eval_batch_size_prompts: int,
        model_kwargs: Mapping[str, Any],
):
    hook_name = sae.cfg.hook_name
    hook_head_index = sae.cfg.hook_head_index

    metric_dict = {}
    if compute_l2_norms:
        metric_dict["l2_norm_in"] = []
        metric_dict["l2_norm_out"] = []
        metric_dict["l2_ratio"] = []
    if compute_sparsity_metrics:
        metric_dict["l0"] = []
        metric_dict["l1"] = []
    if compute_variance_metrics:
        metric_dict["explained_variance"] = []
        metric_dict["mse"] = []

    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)

        # get cache
        _, cache = model.run_with_cache(
            batch_tokens,
            prepend_bos=False,
            names_filter=[hook_name],
            **model_kwargs,
        )

        # we would include hook z, except that we now have base SAE's
        # which will do their own reshaping for hook z.
        has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
        if hook_head_index is not None:
            original_act = cache[hook_name][:, :, hook_head_index]
        elif any(substring in hook_name for substring in has_head_dim_key_substrings):
            original_act = cache[hook_name].flatten(-2, -1)
        else:
            original_act = cache[hook_name]

        # normalise if necessary (necessary in training only, otherwise we should fold the scaling in)
        if activation_store.normalize_activations == "expected_average_only_in":
            original_act = activation_store.apply_norm_scaling_factor(original_act)

        # send the (maybe normalised) activations into the SAE
        sae_feature_activations = sae.encode(original_act.to(sae.device))
        sae_out = sae.decode(sae_feature_activations).to(original_act.device)
        del cache

        if activation_store.normalize_activations == "expected_average_only_in":
            sae_out = activation_store.unscale(sae_out)

        flattened_sae_input = einops.rearrange(original_act, "b ctx d -> (b ctx) d")
        flattened_sae_feature_acts = einops.rearrange(
            sae_feature_activations, "b ctx d -> (b ctx) d"
        )
        flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")

        if compute_l2_norms:
            l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
            l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
            l2_norm_in_for_div = l2_norm_in.clone()
            l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
            l2_norm_ratio = l2_norm_out / l2_norm_in_for_div
            metric_dict["l2_norm_in"].append(l2_norm_in)
            metric_dict["l2_norm_out"].append(l2_norm_out)
            metric_dict["l2_ratio"].append(l2_norm_ratio)

        if compute_sparsity_metrics:
            l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float()
            l1 = flattened_sae_feature_acts.sum(dim=-1)
            metric_dict["l0"].append(l0)
            metric_dict["l1"].append(l1)

        if compute_variance_metrics:
            resid_sum_of_squares = (
                (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
            )
            total_sum_of_squares = (
                (flattened_sae_input - flattened_sae_input.mean(dim=0)).pow(2).sum(-1)
            )
            explained_variance = 1 - resid_sum_of_squares / total_sum_of_squares
            metric_dict["explained_variance"].append(explained_variance)
            metric_dict["mse"].append(resid_sum_of_squares)

    metrics: dict[str, float] = {}
    for metric_name, metric_values in metric_dict.items():
        metrics[f"metrics/{metric_name}"] = torch.stack(metric_values).mean().item()

    return metrics


@torch.no_grad()
def get_recons_loss(
        sae: SAE,
        model: HookedRootModule,
        batch_tokens: torch.Tensor,
        activation_store: ActivationsStore,
        compute_kl: bool,
        compute_ce_loss: bool,
        model_kwargs: Mapping[str, Any] = {},
) -> dict[str, Any]:
    hook_name = sae.cfg.hook_name
    head_index = sae.cfg.hook_head_index

    original_logits, original_ce_loss = model(
        batch_tokens, return_type="both", **model_kwargs
    )

    metrics = {}

    # TODO(tomMcGrath): the rescaling below is a bit of a hack and could probably be tidied up
    def standard_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        activations = sae.decode(sae.encode(activations)).to(activations.dtype)

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)

        return activations.to(original_device)

    def all_head_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        new_activations = sae.decode(sae.encode(activations.flatten(-2, -1))).to(
            activations.dtype
        )

        new_activations = new_activations.reshape(
            activations.shape
        )  # reshape to match original shape

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            new_activations = activation_store.unscale(new_activations)

        return new_activations.to(original_device)

    def single_head_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        new_activations = sae.decode(sae.encode(activations[:, :, head_index])).to(
            activations.dtype
        )
        activations[:, :, head_index] = new_activations

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)
        return activations.to(original_device)

    def standard_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations = torch.zeros_like(activations)
        return activations.to(original_device)

    def single_head_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations[:, :, head_index] = torch.zeros_like(activations[:, :, head_index])
        return activations.to(original_device)

    # we would include hook z, except that we now have base SAE's
    # which will do their own reshaping for hook z.
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_name for substring in has_head_dim_key_substrings):
        if head_index is None:
            replacement_hook = all_head_replacement_hook
            zero_ablate_hook = standard_zero_ablate_hook
        else:
            replacement_hook = single_head_replacement_hook
            zero_ablate_hook = single_head_zero_ablate_hook
    else:
        replacement_hook = standard_replacement_hook
        zero_ablate_hook = standard_zero_ablate_hook

    recons_logits, recons_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, partial(replacement_hook))],
        **model_kwargs,
    )

    zero_abl_logits, zero_abl_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        **model_kwargs,
    )

    def kl(original_logits: torch.Tensor, new_logits: torch.Tensor):
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        log_original_probs = torch.log(original_probs)
        new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
        log_new_probs = torch.log(new_probs)
        kl_div = original_probs * (log_original_probs - log_new_probs)
        kl_div = kl_div.sum(dim=-1)
        return kl_div

    if compute_kl:
        recons_kl_div = kl(original_logits, recons_logits)
        zero_abl_kl_div = kl(original_logits, zero_abl_logits)
        metrics["kl_div_with_sae"] = recons_kl_div
        metrics["kl_div_with_ablation"] = zero_abl_kl_div

    if compute_ce_loss:
        metrics["ce_loss_with_sae"] = recons_ce_loss
        metrics["ce_loss_without_sae"] = original_ce_loss
        metrics["ce_loss_with_ablation"] = zero_abl_ce_loss

    return metrics


def all_loadable_saes() -> list[tuple[str, str, float, float]]:
    all_loadable_saes = []
    saes_directory = get_pretrained_saes_directory()
    for release, lookup in tqdm(saes_directory.items()):
        for sae_name in lookup.saes_map.keys():
            expected_var_explained = lookup.expected_var_explained[sae_name]
            expected_l0 = lookup.expected_l0[sae_name]
            all_loadable_saes.append(
                (release, sae_name, expected_var_explained, expected_l0)
            )

    return all_loadable_saes


def multiple_evals(cfg) -> pd.DataFrame:

    eval_results = []

    # for sae_name, sae_block, _, _ in tqdm(filtered_saes):

    current_model = load_model(model_class_name=cfg.model_class_name,
            model_name=cfg.model_name,
            device=cfg.device,
            n_devices=cfg.n_devices,)

    assert current_model is not None
    for sae_path in tqdm(cfg.sae_paths):
        sae = SAE.load_from_pretrained(sae_path, device="cuda:0")
        for ctx_len in cfg.ctx_lens:
            for dataset in cfg.datasets:
                activation_store = ActivationsStore.from_sae(
                    current_model, sae, context_size=ctx_len, dataset=dataset
                )
                activation_store.shuffle_input_dataset(seed=42)

                eval_metrics = {}
                eval_metrics["sae_id"] = f"{sae_path}-{sae.cfg.hook_name}"
                eval_metrics["context_size"] = ctx_len
                eval_metrics["dataset"] = dataset

                eval_metrics |= run_evals(
                    sae=sae,
                    activation_store=activation_store,
                    model=current_model,
                    eval_config=cfg,
                )

                eval_results.append(eval_metrics)

    return pd.DataFrame(eval_results)


if __name__ == "__main__":
    # Example commands:
    # python sae_lens/evals.py "gpt2-small-res-jb.*" "blocks.8.hook_resid_pre.*" --save_path "gpt2_small_jb_layer8_resid_pre_eval_results.csv"
    # python sae_lens/evals.py "gpt2-small.*" "blocks.8.hook_resid_pre.*" --save_path "gpt2_small_layer8_resid_pre_eval_results.csv"
    # python sae_lens/evals.py "gpt2-small.*" ".*" --save_path "gpt2_small_eval_results.csv"
    # python sae_lens/evals.py "mistral.*" ".*" --save_path "mistral_eval_results.csv"
    cfg = EvalConfig(
        # model
        model_name="/root/data/sae/LLMmodel/XuanYuan-6B-Chat",
        model_class_name="LlamaForCausalLM",
        device="cuda",
        n_devices=4,
        # SAE
        sae_paths=["/root/data/sae/sae_checkpoint/1op8b7lo/final_3072000","/root/data/sae/sae_checkpoint/51tc1n66/final_3072000"],
        # eval config
        num_eval_batches=10,
        eval_batch_size_prompts=8,
        datasets=["/root/data/sae/dataset/FinCorpus3", "/root/data/sae/dataset/FinCorpus2"],
        ctx_lens=[64, 128, 256, 512],# 上下文长度
        save_path="/root/data/sae/evals/unshuffle_shuffle/",
        # Reconstruction metrics
        compute_kl=True, # kl散度
        compute_ce_loss=True,# 交叉熵损失
        compute_l2_norms=True,# l2正则
        # Sparsity and variance metrics
        compute_sparsity_metrics=True,# 特征稀疏度指标
        compute_variance_metrics=True,# 解释方差
    )

    eval_results = multiple_evals(
        cfg
    )
    eval_results.to_csv(cfg.save_path, index=False)
