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
from .llama_general_quant import LlamaForCausalLM
from transformers import LlamaConfig

@MODELS.register_module()
class LlamaForCausalLM_UnropeQuant(HuggingFaceCausalLM):
    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        
        config_keys = {
            "k_bits": None,
            "v_bits": None,
            "k_quant_dim": None,
            "v_quant_dim": None,
            "global_residual_length": None,
            "local_residual_length": None,
            "group_size": -1,
            "rope_scaling": None,
        }

        # 使用字典推导式提取值并设置默认值
        config_values = {key: model_kwargs.pop(key, default) for key, default in config_keys.items()}

        # 设置模型参数的数据类型
        self._set_model_kwargs_torch_dtype(model_kwargs)

        # 从预训练路径加载配置
        config = LlamaConfig.from_pretrained(path)

        # 将提取的值赋值给 config 对象
        for key, value in config_values.items():
            setattr(config, key, value)

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