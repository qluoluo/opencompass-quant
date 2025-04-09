import torch
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import SlurmRunner, LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

from opencompass.models.myModel.kivi.kivi_model import KIVI_LlamaForCausalLM
# from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.models.myModel.hf_strip_model import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM
from opencompass.models.myModel.step_quant.quant_model import LlamaForCausalLM_StepQuant

from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.ruler.ruler_32k_gen_niah_single import ruler_datasets as ruler_datasets_niah_single_3
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets

datasets = []
datasets += ruler_datasets_niah_single_3
# datasets += ruler_datasets

model_path = '/remote-home1/zgliu/models/llama3-8b'
default_model_kwargs = dict(device_map='auto', trust_remote_code=True, torch_dtype=torch.float16, 
                            rope_scaling={"type": "dynamic", "factor": 4.0})
max_seq_len=32*1024
max_out_len=50
run_cfg=dict(num_gpus=1, num_procs=1)
batch_size=1

models = [
    dict(
        type=LlamaForCausalLM_StepQuant,
        abbr='step-4kC-4vC-group32-before',
        model_kwargs=dict(k_bits=4, v_bits=4, global_residual_length=4, local_residual_length=128, group_size=32, k_quant_dim=-1, v_quant_dim=-1),
    ),
    dict(
        type=LlamaForCausalLM_StepQuant,
        abbr='step-4kC-4vT-group32-before',
        model_kwargs=dict(k_bits=4, v_bits=4, global_residual_length=4, local_residual_length=128, group_size=32, k_quant_dim=-1, v_quant_dim=-2),
    ),
    dict(
        type=LlamaForCausalLM_StepQuant,
        abbr='step-4kT-4vT-group32-before',
        model_kwargs=dict(k_bits=4, v_bits=4, global_residual_length=4, local_residual_length=128, group_size=32, k_quant_dim=-2, v_quant_dim=-2),
    ),
    dict(
        type=LlamaForCausalLM_StepQuant,
        abbr='step-4kT-4vT-group256-before',
        model_kwargs=dict(k_bits=4, v_bits=4, global_residual_length=4, local_residual_length=128, group_size=256, k_quant_dim=-2, v_quant_dim=-2),
    ),
    # dict(
    #     type=LlamaForCausalLM_StepQuant,
    #     abbr='step-4kC-4vC-group32-first',
    #     model_kwargs=dict(k_bits=4, v_bits=4, global_residual_length=4, local_residual_length=128, 
    #                       group_size=32, k_quant_dim=-1, v_quant_dim=-1, diff_mode='first'),
    # ),
    # dict(
    #     type=LlamaForCausalLM_StepQuant,
    #     abbr='step-4kC-4vT-group32-first',
    #     model_kwargs=dict(k_bits=4, v_bits=4, global_residual_length=4, local_residual_length=128, 
    #                       group_size=32, k_quant_dim=-1, v_quant_dim=-2, diff_mode='first'),
    # ),
    # dict(
    #     type=LlamaForCausalLM_StepQuant,
    #     abbr='step-4kT-4vT-group32-first',
    #     model_kwargs=dict(k_bits=4, v_bits=4, global_residual_length=4, local_residual_length=128,
    #                       group_size=32, k_quant_dim=-2, v_quant_dim=-2, diff_mode='first'),
    # ),
    # dict(
    #     type=LlamaForCausalLM_StepQuant,
    #     abbr='step-4kC-4vC-group256-first',
    #     model_kwargs=dict(k_bits=4, v_bits=4, global_residual_length=4, local_residual_length=128, 
    #                       group_size=256, k_quant_dim=-1, v_quant_dim=-1, diff_mode='first'),
    # ),
    # dict(
    #     type=LlamaForCausalLM_StepQuant,
    #     abbr='step-4kC-4vT-group256-first',
    #     model_kwargs=dict(k_bits=4, v_bits=4, global_residual_length=4, local_residual_length=128, 
    #                       group_size=256, k_quant_dim=-1, v_quant_dim=-2, diff_mode='first'),
    # ),
    # dict(
    #     type=LlamaForCausalLM_StepQuant,
    #     abbr='step-4kT-4vT-group256-first',
    #     model_kwargs=dict(k_bits=4, v_bits=4, global_residual_length=4, local_residual_length=128,
    #                       group_size=256, k_quant_dim=-2, v_quant_dim=-2, diff_mode='first'),
    # ),
    # dict(
    #     type=LlamaForCausalLM_StepQuant,
    #     abbr='step-2kC-2vT-group32-first',
    #     model_kwargs=dict(k_bits=2, v_bits=2, global_residual_length=4, local_residual_length=128, 
    #                       group_size=32, k_quant_dim=-1, v_quant_dim=-2, diff_mode='first'),
    # ),
    # dict(
    #     type=LlamaForCausalLM_StepQuant,
    #     abbr='step-2kT-2vT-group32-first',
    #     model_kwargs=dict(k_bits=2, v_bits=2, global_residual_length=4, local_residual_length=128,
    #                       group_size=32, k_quant_dim=-2, v_quant_dim=-2, diff_mode='first'),
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