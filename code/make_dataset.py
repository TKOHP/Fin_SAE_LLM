import os

from sae_lens import CacheActivationsRunnerConfig,CacheActivationsRunner
from datasets import load_dataset
import argparse



class make_data:
    def __init__(self,batch_size,traning_step,context_size,n_batches_in_buffer,store_batch_size_prompts,save_path,n_devices):
        self.batch_size=batch_size
        self.traning_step = traning_step
        self.context_size=context_size
        self.n_batches_in_buffer=n_batches_in_buffer
        self.store_batch_size_prompts=store_batch_size_prompts
        self.save_path=save_path
        self.n_devices=n_devices
    def run(self):
        save_path=f"{self.save_path}/bs{self.batch_size}_ts{self.traning_step}_cs{self.context_size}_nbib{self.n_batches_in_buffer}_sbsp{self.store_batch_size_prompts}"
        cfg=CacheActivationsRunnerConfig(
            model_name="/root/data/sae/LLMmodel/XuanYuan-6B-Chat",
            model_class_name="LlamaForCausalLM",
            hook_name="blocks.0.hook_mlp_out",
            context_size=self.context_size,
            d_in=4096,
            training_tokens=self.batch_size*self.traning_step,
            n_batches_in_buffer=self.n_batches_in_buffer,# 和训练的一样
            store_batch_size_prompts=self.store_batch_size_prompts,# 和训练的一样
            new_cached_activations_path=save_path,
            device="cuda",
            n_devices=self.n_devices,
            # ignore
            dataset_path="Duxiaoman-DI/FinCorpus"
        )
        a = CacheActivationsRunner(cfg).run()
        print(a)

def main(args):
    make_data(args.batch_size,args.traning_step,args.context_size,args.n_batches_in_buffer,args.store_batch_size_prompts,args.save_path,args.n_devices).run()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--traning_step', type=int, default=30_00)
    parser.add_argument('--context_size', type=int, default=512)
    parser.add_argument('--n_batches_in_buffer', type=int, default=64)
    parser.add_argument('--store_batch_size_prompts', type=int, default=16)
    parser.add_argument('--save_path', default='/root/data/sae/dataset',
                        help='数据存储根目录，每次存储的文件名都会自动生成，不用改变这个值')
    parser.add_argument('--n_devices', type=int,
                        default=4)

    args = parser.parse_args()
    main(args)
