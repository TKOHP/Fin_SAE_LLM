import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModel, LlamaTokenizer
from transformer_lens import HookedTransformer
from sae_lens import HookedSAETransformer
from sae_lens import SAE,TrainingSAE

from dataclasses import dataclass, field
import sys
from functools import partial
from transformer_lens.utils import test_prompt


# site_packages_path = "/root/data/miniconda3/miniconda3/envs/mxl_vis/lib/python3.10/site-packages"
# # if site_packages_path not in sys.path:
# sys.path.insert(0, site_packages_path)
# sys.path.remove('/home/sae/.local/lib/python3.10/site-packages')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_name_shape_hook_function(activation, hook):
    print(hook.name, activation.shape)


@dataclass
class config:
    model_name: str = "/root/data/sae/LLMmodel/XuanYuan-6B-Chat"
    sae: str = "/root/data/sae/sae_checkpoint/1rs238l9/final_3072000"
    hook_point: str = "blocks.0.hook_mlp_out"
    # feature_index: int = 12
    feature_index: list|None = None
    max_act: int = 30
    steering_strength: float = 3.0
    max_token: int = 1024
    feature_ablation: list | None = None
    use_prompt_ablation: bool = True
    pass1:bool=True
    pass2: bool = True
    pass3: bool = True
    def __post_init__(self):
        pass


class steering:
    def __init__(self, cfg):
        self.model = self.load_model(cfg.model_name)
        print("大模型加载完成")
        self.sae = self.load_sae(cfg.sae)
        self.cfg = cfg
        self.steering_on = False

    def load_model(self, model_name):
        # model_name = model# 本地保存好模型后读取
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        hf_model = LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        # HookedTransformer
        model = HookedSAETransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            device="cuda",
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=self.tokenizer,
            n_devices=4
        )

        # print(model)
        return model

    def load_sae(self, sae):
        encoder = SAE.load_from_pretrained(sae, device="cuda:0")
        # encoder = TrainingSAE.load_from_pretrained(sae, device="cuda:0")
        # for k, v in encoder.named_parameters():
        #     print(f"{k}: {tuple(v.shape)}")
        # encoder.fold_W_dec_norm()
        # print(encoder.W_dec)
        # encoder.W_dec.data[:] = encoder.W_dec / encoder.W_dec.norm(dim=-1, keepdim=True)
        # print(encoder.W_dec)
        # encoder.setup()
        # print(encoder.cfg.normalize_sae_decoder)
        return encoder

    def steering_hook(self, resid_pre, hook):
        coeff = self.cfg.max_act * self.cfg.steering_strength
        # return resid_pre
        if resid_pre.shape[1] == 1:
            return

        # position = self.sae_out.shape[1]
        # print(self.sae_out.shape)
        # print(resid_pre.shape)
        if self.steering_on:
            # using our steering vector and applying the coefficient
            if len(self.cfg.feature_index)==1:
                resid_pre[:, 3:5, :] += coeff * self.steering_vectors
            else:
                for steering_vector in self.steering_vectors:
                    resid_pre[:, :, :] += coeff * steering_vector


    def hooked_generate(self, prompt_batch, fwd_hooks=[]):

        with self.model.hooks(fwd_hooks=fwd_hooks):
            tokenized = self.model.to_tokens(prompt_batch)
            eos_token_id = self.tokenizer.eos_token_id
            result = self.model.generate(input=tokenized, max_new_tokens=cfg.max_token, do_sample=True, temperature=0.7,
                                         top_p=0.95, eos_token_id=eos_token_id, prepend_bos=self.sae.cfg.prepend_bos)
        return result

    def sae_infer(self, sv_prompt):

        sv_logits, cache = self.model.run_with_cache(sv_prompt, prepend_bos=True)
        # print(self.sae.forward(cache[self.cfg.hook_point]))
        tokens = self.model.to_tokens(sv_prompt)
        # print(tokens)
        # get the feature activations from our SAE
        # sv_feature_acts = self.sae.encode(cache[self.cfg.hook_point])
        # sv_feature_acts = self.sae.encode_with_hidden_pre_fn(cache[self.cfg.hook_point])
        # print()
        # # get sae_out
        # self.sae_out = self.sae.decode(sv_feature_acts)

        # print out the top activations, focus on the indices
        # print(torch.topk(sv_feature_acts, 3))

    def run_generate(self, example_prompt):

        editing_hooks = [(self.cfg.hook_point, self.steering_hook)]
        res = self.hooked_generate(example_prompt, editing_hooks)
        # print(f"XuanYuan输出: {res}")
        # Print results, removing the ugly beginning of sequence token
        res_str = self.model.to_string(res[:, 1:])
        print(("\n\n" + "-" * 80 + "\n\n").join(res_str))

    def generate_with_steering(self, prompt):
        self.model.reset_hooks()
        input_ids = self.model.to_tokens(prompt, prepend_bos=self.sae.cfg.prepend_bos)

        steering_vector = self.sae.W_dec[self.cfg.feature_index]

        def steering_l(activations, hook, steering_strength=1.0, steering_vector=None, max_act=1.0):
            # Note if the feature fires anyway, we'd be adding to that here.
            activations[:, :, :] += max_act * steering_strength * steering_vector
            return activations

        steering_hook = partial(
            steering_l,
            steering_vector=steering_vector,
            steering_strength=self.cfg.steering_strength,
            max_act=self.cfg.max_act
        )

        # standard transformerlens syntax for a hook context for generation
        with self.model.hooks(fwd_hooks=[(self.sae.cfg.hook_name, steering_hook)]):
            output = self.model.generate(
                input_ids,
                max_new_tokens=cfg.max_token,
                temperature=0.7,
                top_p=0.95,
                stop_at_eos=True,
                do_sample=True,
                prepend_bos=self.sae.cfg.prepend_bos,
            )

        steered_text = self.model.tokenizer.decode(output[0])
        print(steered_text)

    def steering_with_ablation(self, example_prompt, ablation_features):

        def ablate_feature_hook(feature_activations, hook, feature_ids, position=None):
            # feature_ids=list(range(131072))
            # feature_ids=[]
            if position is None:
                feature_activations[:, :, feature_ids] = 0
            else:
                feature_activations[:, position, feature_ids] = 0
            # print(feature_activations)
            return feature_activations


        ablation_hook = partial(ablate_feature_hook, feature_ids=ablation_features)

        self.model.add_sae(self.sae)
        hook_point = self.sae.cfg.hook_name + '.hook_sae_acts_post'
        # hook_point = self.sae.cfg.hook_name + '.hook_sae_output'
        self.model.add_hook(hook_point, ablation_hook, "fwd")

        tokenized = self.model.to_tokens(example_prompt)
        eos_token_id = self.tokenizer.eos_token_id
        result = self.model.generate(input=tokenized, max_new_tokens=cfg.max_token, do_sample=True, temperature=0.7,
                                     top_p=0.95, eos_token_id=eos_token_id, prepend_bos=self.sae.cfg.prepend_bos)
        res_str = self.model.to_string(result[:, 1:])
        print(("\n\n" + "-" * 80 + "\n\n").join(res_str))



    def feature_index(self, fs_prompt):
        # print("对单个token激活高的特征")
        sv_logits, cache = self.model.run_with_cache(fs_prompt, prepend_bos=True)
        tokens = self.model.to_tokens(fs_prompt)
        # print(tokens)
        # get the feature activations from our SAE
        sv_feature_acts = self.sae.encode(cache[self.cfg.hook_point])
        # get sae_out
        self.sae_out = self.sae.decode(sv_feature_acts)
        # print out the top activations, focus on the indices
        values, indices = torch.topk(sv_feature_acts, 32)
        indices_list = indices[:, 1:, :].tolist()
        feature_indexs = [index for sublist in indices_list for subsublist in sublist for index in subsublist]
        # print("Flattened Indices List:", feature_indexs)
        return feature_indexs

    def run(self, sv_prompt, example_prompt):
        # self.sae_infer(example_prompt)

        # example_prompt="请先复述我的问题，然后回答："+example_prompt
        # print(self.sae_out.shape)
        seps = [" ", "</s>"]
        roles = ["Human", "Assistant"]

        example_prompt = seps[0] + roles[0] + ": " + example_prompt + seps[0] + roles[1] + ":"
        # print(self.sae.W_dec)
        self.steering_vectors = self.sae.W_dec[self.cfg.feature_index]
        self.steering_on = False
        # print("开启引导之前：")
        if self.cfg.pass1:
            self.model.reset_saes()
            self.model.reset_hooks()
            self.run_generate(example_prompt)
        #
        if self.cfg.pass2:
            self.steering_on = True
            print("开启引导之后：")
            self.model.reset_saes()
            self.model.reset_hooks()
            self.run_generate(example_prompt)
        # print("第二个引导方法")
        # self.generate_with_steering(example_prompt)

        # 特征消融
        # answer=""
        # self.sae.use_error_term = False
        if self.cfg.pass3:
            print("开启特征消融之后：")
            self.model.reset_saes()
            self.model.reset_hooks()
            if self.cfg.use_prompt_ablation:
                self.steering_with_ablation(example_prompt, self.feature_index(sv_prompt))
            else:
                self.steering_with_ablation(example_prompt, self.cfg.feature_ablation)

        #
        # print("Test Prompt with feature ablation and error term")
        # self.sae.use_error_term = True
        # self.test_prompt_with_ablation(example_prompt, answer, self.cfg.feature_index)


if __name__ == '__main__':
    import sys

    site_packages_path = "/root/data/miniconda3/miniconda3/envs/mxl_vis/lib/python3.10/site-packages"
    # if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)
    sys.path.remove('/home/sae/.local/lib/python3.10/site-packages')
    set_seed(42)
    cfg = config(
        model_name="/root/data/sae/LLMmodel/XuanYuan-6B-Chat",
        # sae="/root/data/sae/sae_checkpoint/q4tfr87k/final_40960000",
        # sae="/root/data/sae/sae_checkpoint/s78ilkg6/final_40960000",
        sae="/root/data/sae/sae_checkpoint/bll8fob4/final_40960000",
        hook_point="blocks.0.hook_resid_post",
        max_act=10,# 特征的最大激活值
        steering_strength=3,# 强度系数
        feature_index=[93080],
        feature_ablation=
        [22983],
        use_prompt_ablation=True,# 根据prompt搜索相关激活值大的特征，进行消融
        max_token=1024,
        pass1=True,# 引导前是否输出
        pass2=False,# 是否进行正向引导
        pass3=True# 是否进行特征消融
    )
    sv_prompt = "瑞士"
    example_prompt = "简单介绍下瑞士银行"
    steering(cfg).run(sv_prompt, example_prompt)
