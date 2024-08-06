"""
文件名自定义，起个好记的就行，会创建一个和文件名相同的json文件，保存在同一个父目录下。
new_cached_activations_path路径不要以"/"结尾就行，因为要拼接.json

"""
from sae_lens import CacheActivationsRunnerConfig, CacheActivationsRunner
from datasets import load_dataset
import argparse
import os


if __name__ == '__main__':

    batch_size = 1024
    traning_step = 3000
    cfg = CacheActivationsRunnerConfig(
        model_name="D:/project/LLM/myproject/hfl/chinese-llama-2-1.3b",
        model_class_name="LlamaForCausalLM",
        hook_name="blocks.0.hook_mlp_out",
        context_size=512,
        d_in=4096,
        training_tokens=batch_size*traning_step,
        n_batches_in_buffer=8,  # 和训练的一样
        store_batch_size_prompts=8,  # 和训练的一样
        new_cached_activations_path="D:/project/LLM/myproject/activations/2",
        device="cuda",
        n_devices=1,
        # ignore
        dataset_path="Duxiaoman-DI/FinCorpus",
    )
    a = CacheActivationsRunner(cfg).run()
    print(a)
