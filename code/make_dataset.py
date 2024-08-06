
from typing import Any, Literal, Optional, cast
import json
import math
import os
from typing import Tuple
import torch
from tqdm import tqdm

from sae_lens.config import DTYPE_MAP, CacheActivationsRunnerConfig,HfDataset
from sae_lens.load_model import load_model
from sae_lens.training.activations_store import ActivationsStore
class ActivationsStore2(ActivationsStore):
    @torch.no_grad()
    def get_buffer(
            self, n_batches_in_buffer: int, raise_on_epoch_end: bool = False
    ) -> torch.Tensor:
        """
        Loads the next n_batches_in_buffer batches of activations into a tensor and returns half of it.

        The primary purpose here is maintaining a shuffling buffer.

        If raise_on_epoch_end is True, when the dataset it exhausted it will automatically refill the dataset and then raise a StopIteration so that the caller has a chance to react.
        """
        context_size = self.context_size
        batch_size = self.store_batch_size_prompts
        d_in = self.d_in
        total_size = batch_size * n_batches_in_buffer
        num_layers = 1

        if self.cached_activations_path is not None:
            # Load the activations from disk
            buffer_size = total_size * context_size
            # Initialize an empty tensor with an additional dimension for layers
            new_buffer = torch.zeros(
                (buffer_size, num_layers, d_in),
                dtype=self.dtype,  # type: ignore
                device=self.device,
            )
            n_tokens_filled = 0

            # Assume activations for different layers are stored separately and need to be combined
            while n_tokens_filled < buffer_size:
                if not os.path.exists(
                        f"{self.cached_activations_path}/{self.next_cache_idx}.safetensors"
                ):
                    print(
                        "\n\nWarning: Ran out of cached activation files earlier than expected."
                    )
                    print(
                        f"Expected to have {buffer_size} activations, but only found {n_tokens_filled}."
                    )
                    if buffer_size % self.total_training_tokens != 0:
                        print(
                            "This might just be a rounding error — your batch_size * n_batches_in_buffer * context_size is not divisible by your total_training_tokens"
                        )
                    print(f"Returning a buffer of size {n_tokens_filled} instead.")
                    print("\n\n")
                    new_buffer = new_buffer[:n_tokens_filled, ...]
                    return new_buffer

                activations = self.load_buffer(
                    f"{self.cached_activations_path}/{self.next_cache_idx}.safetensors"
                )
                taking_subset_of_file = False
                if n_tokens_filled + activations.shape[0] > buffer_size:
                    activations = activations[: buffer_size - n_tokens_filled, ...]
                    taking_subset_of_file = True

                new_buffer[
                n_tokens_filled: n_tokens_filled + activations.shape[0], ...
                ] = activations

                if taking_subset_of_file:
                    self.next_idx_within_buffer = activations.shape[0]
                else:
                    self.next_cache_idx += 1
                    self.next_idx_within_buffer = 0

                n_tokens_filled += activations.shape[0]

            return new_buffer

        refill_iterator = range(0, batch_size * n_batches_in_buffer, batch_size)
        # Initialize empty tensor buffer of the maximum required size with an additional dimension for layers
        new_buffer = torch.zeros(
            (total_size, context_size, num_layers, d_in),
            dtype=self.dtype,  # type: ignore
            device=self.device,
        )

        for refill_batch_idx_start in refill_iterator:
            # move batch toks to gpu for model
            refill_batch_tokens = self.get_batch_tokens(
                raise_at_epoch_end=raise_on_epoch_end
            ).to(self.device)
            refill_activations = self.get_activations(refill_batch_tokens)
            # move acts back to cpu
            refill_activations.to("cpu")
            new_buffer[
            refill_batch_idx_start: refill_batch_idx_start + batch_size, ...
            ] = refill_activations

            # pbar.update(1)

        new_buffer = new_buffer.reshape(-1, num_layers, d_in)
        new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]

        # every buffer should be normalized:
        if self.normalize_activations == "expected_average_only_in":
            new_buffer = self.apply_norm_scaling_factor(new_buffer)

        return new_buffer


class CacheActivationsRunnerConfig2(CacheActivationsRunnerConfig):
    def __post_init__(self):
        super(CacheActivationsRunnerConfig2,self).__post_init__()

        self.to_json(self.new_cached_activations_path+".json")
    def to_dict(self) -> dict[str, Any]:

        cfg_dict = {
            **self.__dict__,
            # some args may not be serializable by default
            "dtype": str(self.dtype),
            "device": str(self.device),
            "act_store_device": str(self.act_store_device),
        }
        return cfg_dict

    def to_json(self, path: str) -> None:
        json_path=path
        # if not os.path.exists(os.path.dirname(path)):
        #     os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class CacheActivationsRunner:

    def __init__(self,
                 cfg: CacheActivationsRunnerConfig2,
                 override_dataset: HfDataset | None = None
                 ):
        self.cfg = cfg
        self.model = load_model(
            model_class_name=cfg.model_class_name,
            model_name=cfg.model_name,
            device=cfg.device,
            n_devices=cfg.n_devices,
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
        )
        # self.activations_store = ActivationsStore.from_config(
        #     self.model,
        #     cfg,
        # )
        if override_dataset is not None:
            logging.warning(
                f"正在使用覆盖的数据集{override_dataset},{cfg.dataset_path}不起作用"
            )
        self.activations_store = ActivationsStore2.from_config(
            self.model,
            cfg,
            override_dataset=override_dataset,
        )

        self.file_extension = "safetensors"

    def __str__(self):
        """
        Print the number of tokens to be cached.
        Print the number of buffers, and the number of tokens per buffer.
        Print the disk space required to store the activations.

        """

        bytes_per_token = (
            self.cfg.d_in * self.cfg.dtype.itemsize
            if isinstance(self.cfg.dtype, torch.dtype)
            else DTYPE_MAP[self.cfg.dtype].itemsize
        )
        tokens_in_buffer = (
                self.cfg.n_batches_in_buffer
                * self.cfg.store_batch_size_prompts
                * self.cfg.context_size
        )
        total_training_tokens = self.cfg.training_tokens
        total_disk_space_gb = total_training_tokens * bytes_per_token / 10 ** 9

        return (
            f"Activation Cache Runner:\n"
            f"Total training tokens: {total_training_tokens}\n"
            f"Number of buffers: {math.ceil(total_training_tokens / tokens_in_buffer)}\n"
            f"Tokens per buffer: {tokens_in_buffer}\n"
            f"Disk space required: {total_disk_space_gb:.2f} GB\n"
            f"Configuration:\n"
            f"{self.cfg}"
        )

    @torch.no_grad()
    def shuffle_activations(self):
        new_cached_activations_path = self.cfg.new_cached_activations_path
        assert new_cached_activations_path is not None
        if not os.path.exists(new_cached_activations_path):
            raise Exception(
                f"Activations directory ({new_cached_activations_path}) is not exists. Please generate activations first."
            )
        elif len(os.listdir(new_cached_activations_path)) <= 1:
            raise Exception(
                f"Activations directory ({new_cached_activations_path}) is too short to shuffle. Please confirm it larger than 1."
            )
        # get activations
        ##train..
        n_buffers = len(os.listdir(new_cached_activations_path))

        ##shuffle only
        for _ in tqdm(range(self.cfg.n_shuffles_final), desc="Final shuffling"):
            self.shuffle_activations_pairwise(
                new_cached_activations_path,
                buffer_idx_range=(0, n_buffers),
            )

    @torch.no_grad()
    def run(self):
        self.activations_store.track = False
        new_cached_activations_path = self.cfg.new_cached_activations_path

        # if the activations directory exists and has files in it, raise an exception
        assert new_cached_activations_path is not None
        if os.path.exists(new_cached_activations_path):
            if len(os.listdir(new_cached_activations_path)) > 0:
                raise Exception(
                    f"Activations directory ({new_cached_activations_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
                )
        else:
            os.makedirs(new_cached_activations_path)

        print(f"Started caching {self.cfg.training_tokens} activations")
        tokens_per_buffer = (
                self.cfg.store_batch_size_prompts
                * self.cfg.context_size
                * self.cfg.n_batches_in_buffer
        )

        n_buffers = math.ceil(self.cfg.training_tokens / tokens_per_buffer)

        for i in tqdm(range(n_buffers), desc="Caching activations"):
            try:
                buffer = self.activations_store.get_buffer(self.cfg.n_batches_in_buffer)

                self.activations_store.save_buffer(
                    buffer, f"{new_cached_activations_path}/{i}.safetensors"
                )

                del buffer


            except StopIteration:
                print(
                    f"Warning: Ran out of samples while filling the buffer at batch {i} before reaching {n_buffers} batches. No more caching will occur."
                )
                break

    @torch.no_grad()
    def shuffle_activations_pairwise(
            self, datapath: str, buffer_idx_range: Tuple[int, int]
    ):
        """
        Shuffles two buffers on disk.
        """
        assert (
                buffer_idx_range[0] < buffer_idx_range[1] - 1
        ), "buffer_idx_range[0] must be smaller than buffer_idx_range[1] by at least 1"

        buffer_idx1 = torch.randint(
            buffer_idx_range[0], buffer_idx_range[1], (1,)
        ).item()
        buffer_idx2 = torch.randint(
            buffer_idx_range[0], buffer_idx_range[1], (1,)
        ).item()
        while buffer_idx1 == buffer_idx2:  # Make sure they're not the same
            buffer_idx2 = torch.randint(
                buffer_idx_range[0], buffer_idx_range[1], (1,)
            ).item()

        buffer1 = self.activations_store.load_buffer(
            f"{datapath}/{buffer_idx1}.{self.file_extension}"
        )
        buffer2 = self.activations_store.load_buffer(
            f"{datapath}/{buffer_idx2}.{self.file_extension}"
        )
        joint_buffer = torch.cat([buffer1, buffer2])

        # Shuffle them
        joint_buffer = joint_buffer[torch.randperm(joint_buffer.shape[0])]
        shuffled_buffer1 = joint_buffer[: buffer1.shape[0]]
        shuffled_buffer2 = joint_buffer[buffer1.shape[0]:]

        # Save them back
        self.activations_store.save_buffer(
            shuffled_buffer1, f"{datapath}/{buffer_idx1}.{self.file_extension}"
        )
        self.activations_store.save_buffer(
            shuffled_buffer2, f"{datapath}/{buffer_idx2}.{self.file_extension}"
        )


if __name__ == '__main__':
    batch_size = 4096
    traning_step = 10_000
    cfg = CacheActivationsRunnerConfig2(
        model_name="/root/data/sae/LLMmodel/XuanYuan-6B-Chat",
        model_class_name="LlamaForCausalLM",
        hook_name="blocks.0.hook_mlp_out",
        context_size=512,
        d_in=4096,
        training_tokens=batch_size*traning_step,
        n_batches_in_buffer=32,  # 和训练的一样
        store_batch_size_prompts=16,  # 和训练的一样
        new_cached_activations_path="D:/project/LLM/myproject/activations/1",
        device="cuda",
        n_devices=1,
        act_store_device="cpu",
        # ignore
        dataset_path="/root/data/sae/dataset/FinCorpus3",
        # n_shuffles_final=10, #
    )
    # cfg.to_json()
    a = CacheActivationsRunner(cfg)
    ## 生成并保存数据
    a.run()
    ## 进行shuffle
    # a.shuffle_activations()
