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
        type=HuggingFaceCausalLM,
        abbr='llama3-8b-hf',
    ),
    dict(
        type=KIVI_LlamaForCausalLM,
        abbr='llama3-8b-kivi-4k-4v',
        model_kwargs=dict(k_bits=4, v_bits=4, group_size=64, residual_length=128, use_flash=True),
    ),
    dict(
        type=KIVI_LlamaForCausalLM,
        abbr='llama3-8b-kivi-2k-2v',
        model_kwargs=dict(k_bits=2, v_bits=2, group_size=64, residual_length=128, use_flash=True),
    ),
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