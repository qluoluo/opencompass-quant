from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    low_cpu_mem_usage=True,
)

model.generate(
    input_ids=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).cuda(),
    max_new_tokens=10,
)

import transformers.cache_utils