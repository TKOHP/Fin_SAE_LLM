import torch
from datasets import load_dataset
import webbrowser
import os
from transformer_lens import utils, HookedTransformer
from datasets.arrow_dataset import Dataset
from huggingface_hub import hf_hub_download
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, cast
import json
# Library imports
from sae_vis.utils_fns import get_device
from sae_vis.model_fns import AutoEncoder
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.data_config_classes import SaeVisConfig
# from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from sae_lens import SAE
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
from typing import Optional

device = get_device()
torch.set_grad_enabled(False)
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel
import argparse

print(torch.cuda.is_available())


@dataclass
class MyVisConfig:
    model_name: str = "/root/data/sae/LLMmodel/XuanYuan-6B-Chat"
    dataset: str = "/root/data/sae/dataset/FinCorpus3"
    sae: str = "/root/data/sae/sae_checkpoint/pcc1n73m/final_3072000"
    sae_b: str = ""
    hook_point: str = "blocks.0.hook_mlp_out"
    save_html_path: str = "/root/data/sae/mxl_vis/XuanYuan_mb1j2uao"
    is_single: bool = False
    prompt:str="可转换公司债券"
    features:list|range = range(64),
    verbose :bool= True,
    minibatch_size_features :int= 16,
    minibatch_size_tokens :int= 8
    def __post_init__(self):
        if not os.path.exists(self.save_html_path):
            os.makedirs(self.save_html_path)
        self.to_json(self.save_html_path+"/config.json")
    def to_dict(self) -> dict[str, Any]:

        cfg_dict = {
            **self.__dict__,
        }
        cfg_dict["features"]=list(cfg_dict["features"])
        return cfg_dict

    def to_json(self, path: str) -> None:
        # if not os.path.exists(os.path.dirname(path)):
        #     os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class vis:
    def __init__(self, cfg):
        self.cfg=cfg
        self.encoder, self.encoder_B = self.load_sae(cfg.sae, cfg.sae_b)
        self.model = self.load_model(cfg.model_name)

        # .all_tokens = self.get_data(hook_point,save_html_path)
        self.all_tokens = self.get_data()
        self.hook_point = cfg.hook_point

        self.save_html_path = cfg.save_html_path

    def load_sae(self, sae, sae_b):
        encoder = SAE.load_from_pretrained(sae, device="cuda:0")
        if sae_b == "":
            encoder_B = None
        else:
            encoder_B = SAE.load_from_pretrained(sae_b, device="cuda:0")

        for k, v in encoder.named_parameters():
            print(f"{k}: {tuple(v.shape)}")
        return encoder, encoder_B

    def get_data(self):
        SEQ_LEN = 512

        # Load in the data (it's a Dataset object)
        ## 在线读取的话，流式读取，streaming=True
        ## 本地路径读取的话，换为本地路径，可以把streaming=True删除掉。
        # data = load_dataset("NeelNanda/c4-code-20k", split="train")
        data = load_dataset(self.cfg.dataset, split="train[:100]")
        # data = load_dataset("NeelNanda/c4-code-20k", split="train")
        print(type(data))
        # assert isinstance(data, Dataset)

        # Tokenize the data (using a utils function) and shuffle it
        tokenized_data = utils.tokenize_and_concatenate(data, self.model.tokenizer, max_length=SEQ_LEN)  # type: ignore
        tokenized_data = tokenized_data.shuffle(42)

        # Get the tokens as a tensor
        all_tokens = tokenized_data["tokens"]
        assert isinstance(all_tokens, torch.Tensor)
        print(all_tokens.shape)
        return all_tokens

    def load_model(self, model_name):
        # model_name = model# 本地保存好模型后读取
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        hf_model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            device="cuda",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer,
            n_devices=4
        )
        print(model)
        return model

    def make_html(self, hook_point, save_html_path):
        # Specify the hook point you're using, and the features you're analyzing
        sae_vis_config = SaeVisConfig(
            # hook_point=utils.get_act_name("post", 0),
            hook_point=hook_point,
            features=self.cfg.features,
            verbose=self.cfg.verbose,
            minibatch_size_features=self.cfg.minibatch_size_features,
            minibatch_size_tokens=self.cfg.minibatch_size_tokens
        )
        print("开始运行")
        # Gather the feature data
        sae_vis_data = SaeVisData.create(
            encoder=self.encoder,
            encoder_B=self.encoder_B,
            model=self.model,
            tokens=self.all_tokens[: 1024],  # type: ignore
            cfg=sae_vis_config,
        )
        ############prompt为中心的可视化#############
        prompt = "可转换公司债券"
        tokenized_prompt = self.model.tokenizer.tokenize(prompt)
        print(tokenized_prompt)
        # seq_pos=0
        seq_pos = self.model.tokenizer.tokenize(prompt).index("转")
        print(seq_pos)  # type: ignore
        metric = 'act-quantiles'
        save_feature_path = f"{save_html_path}/prompt.html"
        sae_vis_data.save_prompt_centric_vis(
            prompt=prompt,
            filename=save_feature_path,
            seq_pos=seq_pos,  # optional argument, to determine the default option when the page loads
            metric=metric,  # optional argument, to determine the default option when the page loads
        )
        #############feature为中心的可视化###########
        is_single = False
        if is_single == False:
            save_prompt_path = f"{save_html_path}/feature.html"
        # Save as HTML file & open in browser (or not, if in Colab)
        sae_vis_data.save_feature_centric_vis(save_prompt_path, feature_idx=8, is_single=is_single)

        # sae_vis_data.save_json(save_json_path)

    def run(self):
        self.make_html(self.hook_point, self.save_html_path)

if __name__ == '__main__':
    import sys

    site_packages_path = "/root/data/miniconda3/miniconda3/envs/mxl_vis/lib/python3.10/site-packages"
    # if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)
    sys.path.remove('/home/sae/.local/lib/python3.10/site-packages')
    cfg = MyVisConfig(
        model_name="/root/data/sae/LLMmodel/XuanYuan-6B-Chat",
        dataset="/root/data/sae/dataset/FinCorpus3",
        sae="/root/data/sae/sae_checkpoint/51tc1n66/final_3072000",
        sae_b="",
        hook_point="blocks.0.hook_mlp_out",
        save_html_path="/root/data/sae/mxl_vis/XuanYuan_51tc1n66",
        prompt="可转换公司债券",
        features=range(64),
        minibatch_size_features=16,
        minibatch_size_tokens=8
    )
    vis(
        cfg
    ).run()
