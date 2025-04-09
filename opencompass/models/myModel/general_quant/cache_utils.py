from typing import Any, Dict, List, Optional, Tuple, Union
from functools import partial

import torch

from transformers.cache_utils import DynamicCache

from .quant_utils import quantize_tensor, dequantize_tensor

class CustomCache(DynamicCache):
    CustomCache_init = False
    def __init__(self, config) -> None:
        super().__init__()
        if not CustomCache.CustomCache_init:
            CustomCache.CustomCache_init = True
            print("-------------------- CustomCache init ----------------------------------------------------")

        self.key_cache: List[dict] = []
        self.value_cache: List[dict] = []
        # "global", "mid", "local"

        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

        self.kvcache_settings = config.kvcache_settings

        self.key_quant_func = partial(quantize_tensor, nbit=self.kvcache_settings.k_bits, dim=self.kvcache_settings.k_quant_dim)
        self.value_quant_func = partial(quantize_tensor, nbit=self.kvcache_settings.v_bits, dim=self.kvcache_settings.v_quant_dim)
        
        self.key_dequant_func = partial(dequantize_tensor)
        self.value_dequant_func = partial(dequantize_tensor)

        # Independent group sizes for key and value
        self.key_group_size = self.kvcache_settings.key_group_size
        self.value_group_size = self.kvcache_settings.value_group_size

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        selected_key_cache = self.key_cache[layer_idx]
        global_key_cache = selected_key_cache.get("global", None)
        mid_key_cache = selected_key_cache.get("mid", None)
        local_key_cache = selected_key_cache.get("local", None)

        selected_value_cache = self.value_cache[layer_idx]
        global_value_cache = selected_value_cache.get("global", None)
        mid_value_cache = selected_value_cache.get("mid", None)
        local_value_cache = selected_value_cache.get("local", None)
        
        key_cache = self.key_dequant_func(*mid_key_cache)
        value_cache = self.value_dequant_func(*mid_value_cache)

        # Apply group size for key and value independently
        if self.key_group_size > 0:
            key_cache = key_cache.view(*key_cache.shape[:-3], key_cache.shape[-3] * key_cache.shape[-2], key_cache.shape[-1])
        if self.value_group_size > 0:
            value_cache = value_cache.view(*value_cache.shape[:-3], value_cache.shape[-3] * value_cache.shape[-2], value_cache.shape[-1])

        if global_key_cache is not None:
            key_cache = torch.cat([global_key_cache, key_cache], dim=-2)
            value_cache = torch.cat([global_value_cache, value_cache], dim=-2)
        if local_key_cache is not None:
            key_cache = torch.cat([key_cache, local_key_cache], dim=-2)
            value_cache = torch.cat([value_cache, local_value_cache], dim=-2)
        
        return key_cache, value_cache
    
    def __iter__(self):
        raise NotImplementedError
    
    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Get global and local lengths from config
        global_length = self.kvcache_settings.global_residual_length
        local_length = self.kvcache_settings.local_residual_length
        mid_length = self._seen_tokens - global_length - local_length
        key_local_length, value_local_length = local_length, local_length

        # Adjust mid_length based on key and value group sizes
        if self.key_group_size > 0:
            key_local_length += (mid_length % self.key_group_size)
        if self.value_group_size > 0:
            value_local_length += (mid_length % self.value_group_size)
        
        del local_length

        # Initialize dictionaries if not already present
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append({"global": None, "mid": None, "local": None})
            self.value_cache.append({"global": None, "mid": None, "local": None})

        # Get current caches
        current_key_cache = self.key_cache[layer_idx]
        current_value_cache = self.value_cache[layer_idx]

        # 开头部分，直接截取开头部分
        if current_key_cache["global"] is None:
            assert key_states.shape[-2] >= global_length, f"global_length ({global_length}) must be less than key_states.shape[-2] ({key_states.shape[-2]})"

            current_key_cache["global"] = key_states[..., :global_length, :]
            current_value_cache["global"] = value_states[..., :global_length, :]

            key_states = key_states[..., global_length:, :]
            value_states = value_states[..., global_length:, :]

        # 临近部分，先拼接在截取前面的
        if current_key_cache["local"] is not None:
            current_key_cache["local"] = torch.cat([current_key_cache["local"], key_states], dim=-2)
            current_value_cache["local"] = torch.cat([current_value_cache["local"], value_states], dim=-2)
        else:
            current_key_cache["local"] = key_states
            current_value_cache["local"] = value_states

        # 处理key cache local部分中过长的，移动到mid
        if current_key_cache["local"].shape[-2] > key_local_length:
            excess_length = current_key_cache["local"].shape[-2] - key_local_length

            # Only process if excess_length is greater than 0
            if excess_length > 0:
                current_key_cache["local"], excess_key = (
                    current_key_cache["local"][..., -key_local_length:, :],
                    current_key_cache["local"][..., :excess_length, :],
                )

                # Check for grouping, which requires adding a dimension and then restoring it
                concat_dim = -2
                if self.key_group_size > 0:
                    group_size = self.key_group_size
                    num_groups = excess_key.shape[-2] // group_size
                    assert num_groups * group_size == excess_key.shape[-2], \
                        f"excess_key.shape[-2] ({excess_key.shape[-2]}) cannot be divided by group ({group_size})"
                    excess_key = excess_key.view(
                        *excess_key.shape[:-2], num_groups, group_size, excess_key.shape[-1]
                    )
                    concat_dim = -3

                excess_key_quant_data = self.key_quant_func(excess_key)

                if current_key_cache["mid"] is not None:
                    for i, data in enumerate(excess_key_quant_data):
                        current_key_cache["mid"][i] = torch.cat([current_key_cache["mid"][i], excess_key_quant_data[i]], dim=concat_dim)
                else:
                    current_key_cache["mid"] = list(excess_key_quant_data)

        if current_value_cache["local"].shape[-2] > value_local_length:
            excess_length = current_value_cache["local"].shape[-2] - value_local_length

            # Only process if excess_length is greater than 0
            if excess_length > 0:
                current_value_cache["local"], excess_value = (
                    current_value_cache["local"][..., -value_local_length:, :],
                    current_value_cache["local"][..., :excess_length, :],
                )

                concat_dim = -2
                if self.value_group_size > 0:
                    group_size = self.value_group_size
                    num_groups = excess_value.shape[-2] // group_size
                    assert num_groups * group_size == excess_value.shape[-2], \
                        f"excess_value.shape[-2] ({excess_value.shape[-2]}) cannot be divided by group ({group_size})"
                    excess_value = excess_value.view(
                        *excess_value.shape[:-2], num_groups, group_size, excess_value.shape[-1]
                    )
                    concat_dim = -3

                excess_value_quant_data = self.value_quant_func(excess_value)

                if current_value_cache["mid"] is not None:
                    for i, data in enumerate(excess_value_quant_data):
                        current_value_cache["mid"][i] = torch.cat([current_value_cache["mid"][i], excess_value_quant_data[i]], dim=concat_dim)
                else:
                    current_value_cache["mid"] = list(excess_value_quant_data)

        # Update the cache dictionaries
        self.key_cache[layer_idx] = current_key_cache
        self.value_cache[layer_idx] = current_value_cache

        return self[layer_idx]
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
        )
        if is_empty_layer:
            return 0
        
        if self.key_group_size <= 0:
            return self.key_cache[layer_idx]['global'].shape[-2] + self.key_cache[layer_idx]['local'].shape[-2] + self.key_cache[layer_idx]['mid'].shape[-2]
        
        key_mid_length = self.key_cache[layer_idx]['mid'][0].shape[-3] * self.key_group_size if self.key_group_size > 0 else self.key_cache[layer_idx]['mid'].shape[-2]
        # value_mid_length = self.value_cache[layer_idx]['mid'][0].shape[-3] * self.value_group_size if self.value_group_size > 0 else self.value_cache[layer_idx]['mid'].shape[-2]

        return self.key_cache[layer_idx]['global'].shape[-2] + self.key_cache[layer_idx]['local'].shape[-2] + key_mid_length