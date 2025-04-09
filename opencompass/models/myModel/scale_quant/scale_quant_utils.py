import torch
import math

def quantize_tensor(x, nbit, dim):
    # 沿着指定维度找到最大值和最小值
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    
    # 计算scale和zero_point
    scale = (x_max - x_min) / (2**nbit - 1)
    
    zero_point = x_min
    
    # 量化
    x_quantized = ((x - zero_point) / scale).round().clamp(0, 2**nbit - 1).to(dtype=torch.int64)
    
    return x_quantized, scale, zero_point

def dequantize_tensor(x_quantized, scale, zero_point, nbit=None):
    # 还原
    x_dequantized = x_quantized * scale + zero_point
    return x_dequantized

# activate_method_pairs = {
#     'x': {
#         'activated_method': lambda x: x,
#         'deactivated_method': lambda x: x,
#     },
#     'sinx': {
#         'activated_method': lambda x: torch.sin(x),
#         'deactivated_method': lambda x: torch.arcsin(x),
#     },
#     'x^3': {
#         'activated_method': lambda x: torch.pow(x, 3),
#         'deactivated_method': lambda x: torch.pow(x, 1/3),
#     },
#     'x^1/3': {
#         'activated_method': lambda x: torch.pow(x, 1/3),
#         'deactivated_method': lambda x: torch.pow(x, 3)
#     }
# }

def scale_quantize_tensor(x, nbit, dim, activate_method='x'):
    # Step 1: 中心化并缩放到[-π/2, π/2]
    max_val = torch.max(x, dim=dim, keepdim=True).values
    min_val = torch.min(x, dim=dim, keepdim=True).values
    offset = (max_val + min_val) / 2
    x_centered = x - offset
    max_abs = torch.max(torch.abs(x_centered), dim=dim, keepdim=True).values
    x_scaled = x_centered / (max_abs + 1e-8)
    
    # Step 2: 应用sin函数得到[-1, 1]
    if activate_method == 'x':
        x_activated = x_scaled
    elif activate_method == 'arcsinx':
        x_activated = torch.arcsin(x_scaled) / math.pi * 2.0
    elif activate_method == 'sinx':
        x_activated = torch.sin(x_scaled * math.pi / 2.0)
    elif activate_method == 'x^3':
        x_activated = torch.pow(x_scaled, 3)
    elif activate_method == 'x^1/3':
        x_activated = torch.pow(x_scaled, 1/3)
    else:
        raise ValueError("Invalid activate_method.")

    # Step 3: 动态映射到[0, 2^nbit -1]
    quant_range = 2**nbit - 1
    scale_factor = quant_range / 2.0  # 将[-1,1]的跨度2映射到quant_range+1个整数点
    x_mapped = (x_activated + 1) * scale_factor  # [-1,1] -> [0, quant_range]
    x_quantized = torch.round(x_mapped).clamp(0, quant_range)  # 严格限制在整数范围内

    # import ipdb; ipdb.set_trace()

    # 返回量化后的值以及反量化所需的参数
    return x_quantized, offset, max_abs

def scale_dequantize_tensor(x_quantized, offset, max_abs, nbit, activate_method='x'):
    # Step 1: 将量化后的值除以 2^bits
    quant_range = 2**nbit - 1
    scale_factor = quant_range / 2.0  # 将[-1,1]的跨度2映射到quant_range+1个整数点
    x_mapped = x_quantized / scale_factor - 1  # 映射回[-1,1]
    x_mapped = x_mapped.clamp(-1, 1)
    
    # x_arcsin = torch.arcsin(x_mapped.clamp(-1, 1))  # 防止梯度爆炸
    if activate_method == 'x':
        x_deactivated = x_mapped
    elif activate_method == 'arcsinx':
        x_deactivated = torch.sin(x_mapped * math.pi / 2.0)
    elif activate_method == 'sinx':
        x_deactivated = torch.arcsin(x_mapped) / math.pi * 2.0
    elif activate_method == 'x^3':
        x_deactivated = torch.pow(x_mapped, 1/3)
    elif activate_method == 'x^1/3':
        x_deactivated = torch.pow(x_mapped, 3)
    else:
        raise ValueError("Invalid activate_method.")


    x_original_scaled = x_deactivated * max_abs  # 恢复原始缩放比例
    x_dequantized = x_original_scaled + offset  # 恢复中心化偏移
    return x_dequantized


# 示例使用
if __name__ == "__main__":
    tensor = torch.tensor([[1.0, 1.2, 2.3, 2.5, 3.0, 7.7], [3.8, 4.0, 5.0, 6.0, 6.2, 10.6]])
    nbit = 2
    dim = -1

    # 量化
    x_quantized, offset, max_abs = scale_quantize_tensor(tensor, nbit, dim)
    print("\n量化结果:")
    print(x_quantized)
    print(f"{offset.shape=}, {max_abs.shape=}")

    # 反量化
    dequantized_tensor = scale_dequantize_tensor(x_quantized, nbit=nbit, offset=offset, max_abs=max_abs)

    print("\n原始数据:")
    print(tensor)
    print("\n反量化后数据:")
    print(dequantized_tensor)