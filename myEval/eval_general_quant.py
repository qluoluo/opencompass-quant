import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import SlurmRunner, LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

from opencompass.models.myModel.kivi.kivi_model import KIVI_LlamaForCausalLM
# from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.models.myModel.hf_strip_model import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM
from opencompass.models.myModel.general_quant.quant_model import LlamaForCausalLM_GeneralQuant

from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.ruler.ruler_32k_gen_niah_single import ruler_datasets as ruler_datasets_niah_single
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets

datasets = []
# datasets += ruler_datasets_niah_single
datasets += ruler_datasets

model_path = '/remote-home1/zgliu/models/llama3-8b'
default_model_kwargs = dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16, 
                            rope_scaling={"type": "dynamic", "factor": 4.0})
max_seq_len=32*1024
max_out_len=50
run_cfg=dict(num_gpus=1, num_procs=1)
batch_size=1

models = [
    dict(
        type=LlamaForCausalLM_GeneralQuant,
        abbr='llama3-8b-4kTg64-4vC-r128',
        model_kwargs=dict(k_bits=4, v_bits=4, local_residual_length=128, global_residual_length=32,
                          key_group_size=64, k_quant_dim=-2, v_quant_dim=-1),
    ),
    # dict(
    #     type=LlamaForCausalLM_GeneralQuant,
    #     abbr='llama3-8b-4kTg64-4vCg64-r128',
    #     model_kwargs=dict(k_bits=4, v_bits=4, local_residual_length=128, global_residual_length=32,
    #                       key_group_size=64, k_quant_dim=-2, value_group_size=64, v_quant_dim=-1),
    # ),
    dict(
        type=LlamaForCausalLM_GeneralQuant,
        abbr='llama3-8b-2kTg64-2vC-r128',
        model_kwargs=dict(k_bits=2, v_bits=2, local_residual_length=128, global_residual_length=32,
                          key_group_size=64, k_quant_dim=-2, v_quant_dim=-1),
    ),
    # dict(
    #     type=LlamaForCausalLM_GeneralQuant,
    #     abbr='llama3-8b-2kTg64-2vCg64-r128',
    #     model_kwargs=dict(k_bits=2, v_bits=2, local_residual_length=128, global_residual_length=32,
    #                       key_group_size=64, k_quant_dim=-2, value_group_size=64, v_quant_dim=-1),
    # ),
]

for model in models:
    model.update(dict(
        path=model_path,
        model_kwargs=default_model_kwargs | model.get('model_kwargs', {}),
        tokenizer_path=model_path,
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left', trust_remote_code=True),
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        run_cfg=run_cfg,
        batch_size=batch_size,
    ))
    # print(model)

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        task=dict(type=OpenICLInferTask),
        retry=1),
)

##############################################

# python opencompass/run.py opencompass/myEval/eval_general_quant.py -w ./outputs/general_quant --debug