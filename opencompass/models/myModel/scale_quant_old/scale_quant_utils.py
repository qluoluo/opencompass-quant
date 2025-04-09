import torch

def scale_quantize(x, nbit, dim, max_value=1.0, center_offset=True):
    """
    沿着指定维度对数据进行放缩和量化。

    参数:
    x (torch.Tensor): 输入数据
    nbit (int): 量化的比特数
    dim (int): 沿着哪个维度进行放缩和量化
    max_value (float): 放缩后的最大值
    minus_max (bool): 是否先减去使得最大值和最小值的相反数相同的数值

    返回:
    x_quantized (torch.Tensor): 量化后的数据
    scale (torch.Tensor): 放缩系数（用于整体范围调整）
    quant_scale (torch.Tensor): 量化步长（用于具体量化还原）
    zero_point (torch.Tensor): 零点
    shift_value (torch.Tensor): 减去的值
    """
    # 如果选择减去使得最大值和最小值的相反数相同的数值
    if center_offset:
        x_max = x.max(dim=dim, keepdim=True).values
        x_min = x.min(dim=dim, keepdim=True).values
        shift_value = (x_max + x_min) / 2
        x = x - shift_value
    else:
        shift_value = torch.zeros_like(x.mean(dim=dim, keepdim=True))

    # 沿着指定维度找到最大值
    x_max = x.abs().max(dim=dim, keepdim=True).values

    # 计算放缩系数（将数据映射到 [-max_value, max_value]）
    scale = x_max / max_value

    # 放缩数据
    x_scaled = x / scale  # 现在 x_scaled 的范围是 [-max_value, max_value]

    # 沿着指定维度找到最大值和最小值（用于量化）
    x_min = x_scaled.min(dim=dim, keepdim=True).values
    x_max = x_scaled.max(dim=dim, keepdim=True).values

    # 计算量化的 scale 和 zero_point
    quant_scale = (x_max - x_min) / (2**nbit - 1)
    zero_point = x_min.round()  # 零点需要是整数

    # 量化
    x_quantized = ((x_scaled - zero_point) / quant_scale).round().clamp(0, 2**nbit - 1).to(torch.int64)

    return x_quantized, scale, quant_scale, zero_point, shift_value

def scale_dequantize(x_quantized, scale, quant_scale, zero_point, shift_value):
    """
    对量化后的数据进行还原。

    参数:
    x_quantized (torch.Tensor): 量化后的数据
    scale (torch.Tensor): 放缩系数（整体范围）
    quant_scale (torch.Tensor): 量化步长（具体量化步长）
    zero_point (torch.Tensor): 零点
    shift_value (torch.Tensor): 减去的值

    返回:
    x_dequantized (torch.Tensor): 还原后的数据
    """
    # 步骤 1: 还原量化（得到 x_scaled）
    x_scaled = x_quantized * quant_scale + zero_point

    # 步骤 2: 还原放缩（得到去均值后的原始数据）
    x_dequantized = x_scaled * scale

    # 步骤 3: 加回之前减去的值
    x_dequantized = x_dequantized + shift_value

    return x_dequantized

# 示例使用
if __name__ == "__main__":
    tensor = torch.tensor([[1.0, 2.3, 3.0, 7.7], [4.0, 5.0, 6.0, 10.6]])
    nbit = 4
    dim = -1

    # 量化
    quantized_tensor, scale, quant_scale, zero_point, shift_value = scale_quantize(tensor, nbit=nbit, dim=dim, max_value=1.0)

    print(f"输入形状: {tensor.shape}")
    print(f"量化后形状: {quantized_tensor.shape}")
    print(f"放缩系数 scale : {scale}")
    print(f"量化步长 quant_scale : {quant_scale}")
    print(f"零点 zero_point : {zero_point}")
    print(f"减去的值 shift_value: {shift_value}")

    print("\n量化结果:")
    print(quantized_tensor)

    # 还原
    dequantized_tensor = scale_dequantize(quantized_tensor, scale, quant_scale, zero_point, shift_value)

    print("\n原始数据:")
    print(tensor)
    print("\n反量化后数据:")
    print(dequantized_tensor)