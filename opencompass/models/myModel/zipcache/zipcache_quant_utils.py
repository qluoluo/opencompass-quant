import torch

def zipcache_quantize(x, nbit, scale_dim, quant_dim, scale_method='max_sqrt'):
    c = torch.max(torch.abs(x), dim=scale_dim, keepdim=True).values
    if scale_method == 'max_sqrt':
        c = torch.sqrt(c)
    elif scale_method == 'max':
        c = c
    else:
        raise ValueError('scale_method not supported')
    x = x / c

    x_min = x.min(dim=quant_dim, keepdim=True).values
    x_max = x.max(dim=quant_dim, keepdim=True).values
    
    # 计算scale和zero_point
    scale = (x_max - x_min) / (2**nbit - 1)
    
    zero_point = x_min
    x_quantized = ((x - zero_point) / scale).round().clamp(0, 2**nbit - 1).to(dtype=torch.int64)

    return x_quantized, c, scale, zero_point

def zipcache_dequantize(x_quantized, c, scale, zero_point):
    # 反量化计算
    x_dequantized = x_quantized * scale + zero_point
    # 还原归一化操作
    x_recovered = x_dequantized * c
    return x_recovered

if __name__ == "__main__":
    tensor = torch.tensor([[1.0, 2.3, 3.0, 7.7], [4.0, 5.0, 6.0, 10.6]])
    nbit = 4
    scale_dim = -2
    quant_dim = -1

    # 量化
    quantized_tensor, c, scale, zero_point = zipcache_quantize(tensor, nbit=nbit, scale_dim=scale_dim, quant_dim=quant_dim)

    print(f"输入形状: {tensor.shape}")
    print(f"量化后形状: {quantized_tensor.shape}")
    print(f"放缩系数 c : {c}")
    print(f"量化步长 scale : {scale}")
    print(f"零点 zero_point : {zero_point}")

    print("\n量化结果:")
    print(quantized_tensor)

    # 还原
    dequantized_tensor = zipcache_dequantize(quantized_tensor, c, scale, zero_point)

    print("\n原始数据:")
    print(tensor)
    print("\n反量化后数据:")
    print(dequantized_tensor)