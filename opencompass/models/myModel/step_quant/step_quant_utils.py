import torch
import math

def compute_diff(x, diff_mode='before'):
    # 提取第一个 feature 长度的 tensor
    first_feature = x[..., 0:1, :]  # 保持第一个 feature 不变
    
    if diff_mode == 'before':
        diff = x[..., 1:, :] - x[..., :-1, :]
    elif diff_mode == 'first':
        # 保持维度一致
        diff = x[..., 1:, :] - x[..., 0:1, :]
    else:
        raise ValueError("Invalid mode. Choose 'before' or 'first'.")
    
    diff_tensor = torch.cat((first_feature, diff), dim=-2)
    return diff_tensor

def restore_from_diff(diff_tensor: torch.Tensor, mode='before'):
    # 提取第一个 feature 长度的 tensor
    first_feature = diff_tensor[..., 0:1, :]  # 第一个 feature 是原始值
    
    if mode == 'before':
        # 逐步还原其余部分
        restored_tensor = diff_tensor.cumsum(dim=-2)
    elif mode == 'first':
        # 直接使用第一个 feature 进行还原
        restored_tensor = first_feature + diff_tensor[..., 1:, :]
        # 将第一个 feature 拼接到还原结果的开头
        restored_tensor = torch.cat((first_feature, restored_tensor), dim=-2)
    else:
        raise ValueError("Invalid mode. Choose 'before' or 'first'.")
    
    return restored_tensor


def quantize(x, nbit, dim):
    # 沿着指定维度找到最大值和最小值
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    
    # 计算scale和zero_point
    scale = (x_max - x_min) / (2**nbit - 1)
    
    zero_point = x_min
    
    # 量化
    x_quantized = ((x - zero_point) / scale).round().clamp(0, 2**nbit - 1).to(dtype=torch.int64)
    
    return x_quantized, scale, zero_point

def dequantize(x_quantized, scale, zero_point):
    # 还原
    x_dequantized = x_quantized * scale + zero_point
    return x_dequantized

def step_quant(x, nbit, dim, global_residual_length=0, local_residual_length=128, group_size=0, 
               quantize_drop_first=True, diff_mode='before'):
    
    seq_len = x.shape[-2]
    mid_len = seq_len - local_residual_length - global_residual_length
    
    # 如果分组，调整 local_residual_length 以处理无法整除的情况
    if group_size > 0:
        local_residual_length += (mid_len % group_size)
    
    # 提取全局残差和局部残差
    x_global = None
    x_local = None
    if global_residual_length > 0:
        x_global = x[..., :global_residual_length, :]
        x = x[..., global_residual_length:, :]
    if local_residual_length > 0:
        x_local = x[..., -local_residual_length:, :]
        x = x[..., :-local_residual_length, :]
    
    # 分组逻辑
    if group_size > 0:
        num_groups = x.shape[-2] // group_size
        assert num_groups * group_size == x.shape[-2], f"x.shape[-2] ({x.shape[-2]}) cannot be divided by group ({group_size})"
        x = x.view(*x.shape[:-2], num_groups, group_size, x.shape[-1])
    
    # 对每个组进行差分操作
    x_diff = compute_diff(x, diff_mode=diff_mode)
    
    # 量化（除了第一个特征，除非 quantize_all 为 True）
    if quantize_drop_first:
        first_feature = x_diff[..., 0:1, :]  # 每组第一个 feature
        x_quantized, scale, zero_point = quantize(x_diff[..., 1:, :], nbit=nbit, dim=dim)
    else:
        first_feature = None
        x_quantized, scale, zero_point = quantize(x_diff, nbit=nbit, dim=dim)
    
    # 返回 global、local、每组第一个 feature 和量化后的差分
    return x_global, x_local, first_feature, x_quantized, scale, zero_point


def step_dequant(x_global, x_local, first_feature, x_quantized, scale, zero_point, group_size=0,
                 quantize_drop_first=True, diff_mode='before'):
    
    # 反量化差分部分
    x_diff_dequant = dequantize(x_quantized, scale, zero_point)
    
    # 如果保留了每组第一个 feature，将其与反量化后的差分拼接
    if quantize_drop_first and first_feature is not None:
        x_diff_dequant = torch.cat((first_feature, x_diff_dequant), dim=-2)
    
    # 还原差分
    x_restored = restore_from_diff(x_diff_dequant)
    
    # 还原分组
    if group_size > 0:
        x_restored = x_restored.view(*x_restored.shape[:-3], -1, x_restored.shape[-1])
    
    # 合并全局和局部残差
    if x_global is not None:
        x_restored = torch.cat((x_global, x_restored), dim=-2)
    if x_local is not None:
        x_restored = torch.cat((x_restored, x_local), dim=-2)
    
    return x_restored


def vanlidate_diff():
    # 构造一个随机的 3D 张量 x，形状为 (batch_size, time_steps, features)
    batch_size = 3
    time_steps = 1024
    features = 128
    group_size = 23
    diff_mode = 'first'

    x = torch.randn(batch_size, time_steps, features)

    # 计算差分张量
    diff_tensor = compute_diff(x, diff_mode)

    # 从差分张量还原原始张量
    restored_tensor = restore_from_diff(diff_tensor, diff_mode)

    # 检查还原的张量是否与原始张量相等
    if torch.allclose(x, restored_tensor):
        print("Test passed: The restored tensor matches the original tensor.")
    else:
        print("Test failed: The restored tensor does not match the original tensor.")
        print(f"{torch.max(torch.abs(x - restored_tensor))}")

def vanlidate_step_quant():
    # 构造一个随机的 3D 张量 x，形状为 (batch_size, time_steps, features)
    batch_size = 3
    time_steps = 1024
    features = 128
    group_size = 23
    nbit = 4
    dim = -2

    x = torch.randn(batch_size, time_steps, features)
    quant_data = step_quant(x, group_size=group_size, nbit=nbit, dim=dim)
    print(f"{quant_data[3]=}")
    x_restored = step_dequant(*quant_data, group_size=group_size)

    # print(f"{x=}, {x_restored=}")
    # print(f"{torch.max(torch.abs(x - x_restored))}")
    print(f"{x-x_restored=}")

# 示例使用
if __name__ == "__main__":
    pass

    vanlidate_diff()
    # vanlidate_step_quant()