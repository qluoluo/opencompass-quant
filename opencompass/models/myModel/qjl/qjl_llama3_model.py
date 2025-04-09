import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

# from .huggingface import HuggingFaceCausalLM
# from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.models.myModel.hf_strip_model import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM
from .QJL.models.llama3_qjl import LlamaForCausalLM_QJL
from .QJL.models.llama3_utils_qjl import QJLSketch
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM

@MODELS.register_module()
class QJL_LlamaForCausalLM(HuggingFaceCausalLM):
    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        # from transformers import AutoModelForCausalLM
        default_qjl_kwargs = {
            'key_quantization_bits': 256,
            'key_quantization_bits_initial_layers': 512,
            'initial_layers_count': 15,
            'outlier_count_general': 8,
            'outlier_count_initial_layers': 8,
            'value_quantization_bits': 2,
            'group_size': 32,
            'buffer_size': 128,
        }
        default_qjl_kwargs = {k: model_kwargs.pop(k, default_qjl_kwargs[k]) for k in default_qjl_kwargs}

        self._set_model_kwargs_torch_dtype(model_kwargs)

        config = LlamaConfig.from_pretrained(path)
        config.attention_dropout = 0.0
        for k, v in default_qjl_kwargs.items():
            setattr(config, k, v)
        
        device = 'cuda'
        generator = torch.Generator(device=torch.device(device))

        config.qjl = QJLSketch(dim=(128, config.key_quantization_bits), dim_outlier=256, rot=True, rng=generator)
        config.qjl_initial_layers = QJLSketch(dim=(128, config.key_quantization_bits_initial_layers), dim_outlier=128,
                                                rot=True,
                                                rng=generator)

        config.use_flash = True

        self.model = LlamaForCausalLM_QJL.from_pretrained(
            pretrained_model_name_or_path=path,
            config=config,
            cache_dir=None,
            low_cpu_mem_usage=True,
            **model_kwargs
        )
        
        self.model.eval()
        self.model.generation_config.do_sample = False