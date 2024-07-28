import torch
from datasets import load_dataset
import webbrowser
import os
from transformer_lens import utils, HookedTransformer
from datasets.arrow_dataset import Dataset
from huggingface_hub import hf_hub_download
import time

# Library imports
from sae_vis.utils_fns import get_device
from sae_vis.model_fns import AutoEncoder
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.data_config_classes import SaeVisConfig
# from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from sae_lens import SAE

device = get_device()
torch.set_grad_enabled(False)
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel
import argparse

print(torch.cuda.is_available())


class vis:
    def __init__(self, model_name,
                 sae,
                 sae_b,
                 hook_point,
                 save_html_path):

        self.encoder, self.encoder_B = self.load_sae(sae,sae_b)
        self.model = self.load_model(model_name)
        self.all_tokens = self.get_data(hook_point,save_html_path)
        model_name_spilt=model_name.split("/")[-1].replace("-","_")
        sae_split = sae.split("/")[-1]
        sae_b_spilt = sae_b.split("/")[-1]
        save_html_path = f"{save_html_path}/{model_name_spilt}_a_{sae_split}_b_{sae_b_spilt}_{hook_point}.html"

    def load_sae(self,sae,sae_b):
        encoder = SAE.load_from_pretrained(sae)
        if sae_b=="":
            encoder_B = None
        else:
            encoder_B = SAE.load_from_pretrained(sae_b)

        for k, v in encoder.named_parameters():
            print(f"{k}: {tuple(v.shape)}")
        return encoder, encoder_B


    def get_data(self):
        SEQ_LEN = 512

        # Load in the data (it's a Dataset object)
        ## 在线读取的话，流式读取，streaming=True
        ## 本地路径读取的话，换为本地路径，可以把streaming=True删除掉。
        data = load_dataset("NeelNanda/c4-code-20k", split="train")
        # data = load_dataset("Duxiaoman-DI/FinCorpus", split="train[:100]")
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


    def load_model(self,model_name):
        # model_name = model# 本地保存好模型后读取
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        hf_model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            device="cpu",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer,
        )
        model = model.to(device)
        return model


    def run(self,hook_point,save_html_path):
        # Specify the hook point you're using, and the features you're analyzing
        sae_vis_config = SaeVisConfig(
            # hook_point=utils.get_act_name("post", 0),
            hook_point=hook_point,
            features=range(64),
            verbose=True,
        )
        # Gather the feature data
        sae_vis_data = SaeVisData.create(
            encoder=self.encoder,
            # encoder_B=self.encoder_B,
            model=self.model,
            tokens=self.all_tokens[: 4096],  # type: ignore
            cfg=sae_vis_config,
        )

        # Save as HTML file & open in browser (or not, if in Colab)
        sae_vis_data.save_feature_centric_vis(save_html_path, feature_idx=8)


def main(args):
    vis(
        args.model_name,
        args.sae,
        args.sae_b,
        args.hook_point,
        args.save_html_path,
    ).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--model_name', default="/root/data/sae/LLMmodel/XuanYuan-6B-Chat", help="大模型的位置")
    parser.add_argument('--sae', default="/root/data/sae/sae_checkpoint/2eizws4q", help="sae的checkpoint路径")
    parser.add_argument('--sae_b', default="/root/data/sae/sae_checkpoint/2eizws4q",help="sae的checkpoint路径")
    parser.add_argument('--hook_point', default="blocks.0.hook_mlp_out", help="在MLP的哪一层")
    parser.add_argument('--save_html_path', default="/root/data/sae/vis_html")
    args = parser.parse_args()
    main(args)
