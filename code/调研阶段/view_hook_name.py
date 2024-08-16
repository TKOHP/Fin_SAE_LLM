
from transformers import LlamaTokenizer,LlamaForCausalLM
from transformer_lens import HookedTransformer


def print_name_shape_hook_function(activation, hook):
    print(hook.name, activation.shape)
if __name__ == '__main__':
    model_name="/root/data/sae/LLMmodel/XuanYuan-6B-Chat"
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
    hook_points = model.hook_dict.keys()
    print("List of hook points in the model:")
    # for hook in hook_points:
    #     print(hook)
    test_prompt = "你好"
    print("Num tokens:", len(model.to_tokens(test_prompt)[0]))
    not_in_late_block_filter = lambda name: name.startswith("blocks.0.") or not name.startswith("blocks")

    model.run_with_hooks(
        test_prompt,
        return_type=None,
        fwd_hooks=[(not_in_late_block_filter, print_name_shape_hook_function)],
    )
