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

# from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.models.myModel.hf_strip_model import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM
from .llama_scale_quant import LlamaForCausalLM
from transformers import LlamaConfig

@MODELS.register_module()
class LlamaForCausalLM_ScaleQuant(HuggingFaceCausalLM):
    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        # from transformers import AutoModelForCausalLM
        k_bits, v_bits, residual_length = model_kwargs.pop("k_bits"), model_kwargs.pop("v_bits"), model_kwargs.pop("residual_length")
        k_quant_dim, v_quant_dim = model_kwargs.pop("k_quant_dim"), model_kwargs.pop("v_quant_dim")
        group_size = model_kwargs.pop("group_size", -1)
        center_offset = model_kwargs.pop("center_offset", True)

        rope_scaling = model_kwargs.pop("rope_scaling", None)

        self._set_model_kwargs_torch_dtype(model_kwargs)

        config = LlamaConfig.from_pretrained(path)
        config.k_bits = k_bits
        config.v_bits = v_bits
        config.k_quant_dim = k_quant_dim
        config.v_quant_dim = v_quant_dim
        config.group_size = group_size
        config.residual_length = residual_length # the number of recent fp16 tokens
        config.rope_scaling = rope_scaling
        config.center_offset = center_offset

        CACHE_DIR = './cache'

        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path,
            config=config,
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True,
            **model_kwargs
        )

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        
        self.model.eval()
        self.model.generation_config.do_sample = False