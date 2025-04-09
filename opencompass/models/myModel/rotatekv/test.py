import os, torch
from .modeling_llama_RotateKV import LlamaForCausalLM_RotateKV
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaConfig
from .RotateKV.fuse_weights import fuse_weights

model_path = "/remote-home1/zgliu/models/llama3-8b"
model = LlamaForCausalLM_RotateKV.from_pretrained(model_path)
config = LlamaConfig.from_pretrained(model_path)

config.attn_implementation = "flash_attention_2"
config.head_group_num = 4
config.fuse_weights = True        
config.Grouped_Head_Key_Rotation = True
config.Outlier_Aware_Key_Rotation = True
config.Attention_Sink_Aware = True    
config.k_bits = 2
config.v_bits = 2 
config.k_clip_ratio = 0.8
config.v_clip_ratio = 0.8

model = LlamaForCausalLM_RotateKV.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,attn_implementation="flash_attention_2", use_safetensors=True,
    config=config
)
fuse_weights(model)
model.eval()