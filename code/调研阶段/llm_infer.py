import torch
from transformers import LlamaForCausalLM, AutoTokenizer,AutoModel
import pandas as pd
import json
from peft import PeftModel

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class xuanyuan:
    def __init__(self):
        model_name_or_path = "/root/data/sae/LLMmodel/XuanYuan-6B-Chat"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
        self.model.eval()
    def infer(self,content):
        seps = [" ", "</s>"]
        roles = ["Human", "Assistant"]

        prompt = seps[0] + roles[0] + ": " + content + seps[0] + roles[1] + ":"
        print(f"输入: {content}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        eos_token_id = self.tokenizer.eos_token_id
        outputs = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.95,eos_token_id =eos_token_id)
        outputs = self.tokenizer.decode(outputs.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(f"XuanYuan输出: {outputs}")

class fingpt:
    def __init__(self):
        base_model_path = "/root/data/sae/LLMmodel/base_models/chatglm2-6b"
        base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True, load_in_8bit=False, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        # Load Models
        peft_model_path = "/root/data/sae/LLMmodel/fingpt-mt_chatglm2-6b_lora"
        self.model = PeftModel.from_pretrained(base_model, peft_model_path)
        self.model=self.model.eval()

    def encode(self,x):
        ids = self.tokenizer.encode(x, max_length=1024, return_tensors="np", padding="max_length")
        input_ids = torch.LongTensor([ids])
        return input_ids
    def infer(self,content):
        # Make prompts
        # Please change the news to the news you want
        # news = 'A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertiser'
        template = """News: '''{}'''
        Answer:
        """
        prompt = template.format(content)
        # # response, history = self.model.chat(self.tokenizer, "你好", history=[])
        # # print(response)
        # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        # generation_config = {
        #     'max_new_tokens': 200,
        #     'do_sample': False,
        #     'num_beams': 8,
        #     'early_stopping': True,
        # }
        # outputs = self.model.generate(input_ids, **generation_config)
        # rets = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        content = "某人向银行借款10万元，借款期限是36个月，年利率是6%，还款方式是等额本金，则第二个月的利息是多少，列出计算公式和答案"
        eos_token_id = self.tokenizer.eos_token_id
        other={"eos_token_id":eos_token_id}
        response, history = self.model.chat(self.tokenizer, content, history=[],**other)
        print("Fingpt输出:")
        print(response)
        # print(outputs)
class fingptv:
    def __init__(self):
        base_model_path = "/root/data/sae/LLMmodel/base_models/chatglm2-6b"
        #self.base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True, load_in_8bit=False, device_map='auto').half().cuda()
        self.base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True).half().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        peft_model_path = "/root/data/sae/LLMmodel/fingpt-mt_chatglm2-6b_lora"
        self.model = PeftModel.from_pretrained(self.base_model, peft_model_path)
        self.model=self.model.eval()

    def encode(self,x):
        ids = self.tokenizer.encode(x, max_length=1024, return_tensors="np", padding="max_length")
        input_ids = torch.LongTensor([ids])
        return input_ids
    def infer(self,content):
        # Make prompts
        # Please change the news to the news you want
        # news = 'A tinyurl link takes users to a scamming site promising that users can earn thousands of dollars by becoming a Google ( NASDAQ : GOOG ) Cash advertiser'
        template = """News: '''{}'''

        Instruction: Please 'ONLY' output 'one' sentiment of all the above News from {{ Severely Positive / Moderately Positive / Mildly Positive / Neutral / Mildly Negative / Moderately Negative / Severely Negative }} without other words.

        Answer:
        """
        prompt = template.format(content)

        tokens = self.tokenizer(prompt, return_tensors='pt', padding=True)
        tokens = tokens.to(self.model.device)
        generation_config = {
            'max_new_tokens': 100,
            'do_sample': False,
            'num_beams': 4,
            'early_stopping': True,
            'pad_token_id': self.tokenizer.eos_token_id
        }
        with torch.no_grad():
            resp, history = self.base_model.chat(self.tokenizer, "你好", history=[])
            res = self.model.generate(**tokens, **generation_config)
        torch.cuda.empty_cache()
        print(resp)
        res_sentences = [self.tokenizer.decode(i, skip_special_tokens=True) for i in res]
        out_text = [o.split("Answer:")[1] for o in res_sentences]
        sentiment = out_text[0].strip()
        print(sentiment)

if __name__ == '__main__':
    set_seed(52)
    content="某人向银行借款10万元，借款期限是36个月，年利率是6%，还款方式是等额本金，则第二个月的利息是多少，列出计算公式和答案"
    # content="Why can't I read your output?"
    # a=xuanyuan()
    b = fingpt()

    # a.infer(content)
    print("----------------------------------------------------")
    b.infer(content)