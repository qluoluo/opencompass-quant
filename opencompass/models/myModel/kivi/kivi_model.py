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
from .llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM

@MODELS.register_module()
class KIVI_LlamaForCausalLM(HuggingFaceCausalLM):
    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        # from transformers import AutoModelForCausalLM
        k_bits, v_bits, group_size, residual_length = model_kwargs.pop("k_bits"), model_kwargs.pop("v_bits"), model_kwargs.pop("group_size"), model_kwargs.pop("residual_length")
        use_flash = model_kwargs.pop("use_flash", True)
        rope_scaling = model_kwargs.pop('rope_scaling', None)

        self._set_model_kwargs_torch_dtype(model_kwargs)

        config = LlamaConfig.from_pretrained(path)
        config.k_bits = k_bits # current support 2/4 bit for KV Cache
        config.v_bits = v_bits # current support 2/4 bit for KV Cache
        config.group_size = group_size
        config.residual_length = residual_length # the number of recent fp16 tokens
        config.use_flash = use_flash
        config.rope_scaling = rope_scaling
        CACHE_DIR = './cache'

        self.model = LlamaForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=path,
            config=config,
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True,
            **model_kwargs
        )

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        
        self.model.eval()
        self.model.generation_config.do_sample = False