import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# from transformers import LlamaConfig, LlamaForCausalLM
from modeling_llama import LlamaConfig, LlamaForCausalLM
import torch
from transformers import AutoTokenizer

MODEL_PATH = '/remote-home1/share/models/llama3_2_hf/Llama-3.2-3B/'

config = LlamaConfig.from_pretrained(MODEL_PATH)
config.replaced_layers = [0,1,2]
config.target_layer_type = 'performer'

print("加载模型...")
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

model.eval()
model.generation_config.do_sample = False

# 生成文本
input_text = "你" * 32 * 1024
inputs = tokenizer(input_text, return_tensors='pt')
print(f"input_length = {inputs['input_ids'].shape[1]}")

# 移动输入到与模型相同的设备
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 生成配置
output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id  # 添加pad_token_id避免警告
)

# 解码输出
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)