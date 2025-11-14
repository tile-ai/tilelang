import tilelang
import torch
import tilelang.language as T

# n = 2 ** 25
B = 8
t = 2**11
D = 128
k = torch.randn(B,t,D, dtype=torch.float32, device='cuda')
s = torch.softmax(torch.randn(B,t,3, dtype=torch.float32, device='cuda'),dim=-1)

def shift_with_zeros(x, shift, dim):
    """
    沿指定维度平移张量，移出去的部分用 0 填充
    x:   输入张量
    shift: 正数表示向后（高索引）移动，负数表示向前（低索引）移动
    dim:  平移的维度
    """
    if shift == 0:
        return x
    # 记录张量形状
    zeros_shape = list(x.shape)
    zeros_shape[dim] = abs(shift)
    zeros = torch.zeros(zeros_shape, dtype=x.dtype, device=x.device)

    if shift > 0:
        # 向后移动
        return torch.cat([zeros, x.narrow(dim, 0, x.shape[dim] - shift)], dim=dim)
    else:
        # 向前移动
        shift = -shift
        return torch.cat([x.narrow(dim, shift, x.shape[dim] - shift), zeros], dim=dim)

def make_first_recurrent(k, s):
    """
    k: [b, h, t, d]
    s: [b, h, t, 3]
    非循环位移版本：torch.roll 改为 shift_with_zeros
    """
    b, h, t, d = k.shape
    device = k.device
    dtype = k.dtype
    # 初始化 S（不含时间维度）
    S = torch.zeros((b, h, d, d), dtype=dtype, device=device)
    o = []
    for i in range(t):
        # 保存当前 time step 的 S[:, :, 0] （加一个时间维）
        o.append(S[:, :, 0].unsqueeze(2))
        # 左右平移（补零）
        S_left  = shift_with_zeros(S, 1, dim=2)   # j-1
        S_right = shift_with_zeros(S, -1, dim=2)  # j+1
        # 取权重并广播
        w0 = s[:, :, i, 0].unsqueeze(-1).unsqueeze(-1)   # [b,h,1,1]
        w1 = s[:, :, i, 1].unsqueeze(-1).unsqueeze(-1)
        w2 = s[:, :, i, 2].unsqueeze(-1).unsqueeze(-1)
        # 更新 S
        S = S_left * w0 + S * w1 + S_right * w2
        # 更新 S 的第 0 列
        S[:, :, 0] = S[:, :, 0] + w0.squeeze(-1) * k[:, :, i]
    return torch.cat(o, dim=2)
block_size = 32
num_block = t // block_size
o_torch = torch.cat([ make_first_recurrent(k[:,i*block_size: (i+1)* block_size].unsqueeze(1),s[:,i*block_size: (i+1)* block_size].unsqueeze(1))for i in range(num_block)],dim=2).unsqueeze(1)

@tilelang.jit
def inner_chunk_recurrent_fwd_init0(b,t,d,blk_t=block_size) -> tilelang.JITKernel:

    @T.prim_func
    def inner_chunk_recurrent_fwd_init0_(
        S: T.Tensor((b, t//blk_t, d, d), 'float32'),
        k: T.Tensor((b, t, d), 'float32'),
        s: T.Tensor((b, t, 3), 'float32'),
        o: T.Tensor((b, t, d), 'float32'),
    ):
        
        with T.Kernel(b * d,T.ceildiv(t, blk_t)) as (i_bd, i_t):
            i_b = i_bd // d
            i_d = i_bd % d
            S_temp = T.alloc_fragment(d, 'float32') 
            S_down = T.alloc_fragment(d, 'float32') 
            S_up   = T.alloc_fragment(d, 'float32') 
            S_mid  = T.alloc_fragment(d, 'float32') 
            for i0_d in T.Parallel(d):
                S_temp[i0_d] = 0 
                S_down[i0_d] = 0
                S_up[i0_d]   = 0
                S_mid[i0_d]   = 0
            for i0_t in T.serial(blk_t):
                t_local = i0_t*blk_t + i0_t
                #先存第一行也就是栈顶，到输出的o里面
                o[i_b,t_local,i_d] = S_temp[0]
                #再做三对角，实际上也就是相邻行的加权求和
                down =  s[i_b,t_local,0]
                mid = s[i_b,t_local,1]
                up = s[i_b,t_local,2]
                for i0_d in T.Parallel(d-1):
                    S_down[i0_d + 1] = S_temp[i0_d] * down
                for i0_d in T.Parallel(d-1):       
                    S_up[i0_d] = S_temp[i0_d + 1] * up 
                for i0_d in T.Parallel(d):       
                    S_mid[i0_d] = S_temp[i0_d] * mid
                S_down[0] = 0
                S_up[d-1] = 0
                for i0_d in T.Parallel(d):       
                    S_temp[i0_d] += S_mid[i0_d]
                    S_temp[i0_d] += S_down[i0_d]
                    S_temp[i0_d] += S_up[i0_d]
                #往栈顶写入当前的k
                S_temp[0] += down * k[i_b,t_local,i_d]
            # 存储当前block最终的状态S,留作未来计算
            for i0_d in T.Parallel(d):
                S[i_b,i_t,i0_d,i_d] = S_temp[i0_d]
    return inner_chunk_recurrent_fwd_init0_

# 这个参数是可以灵活配置的
for blk_t in [32,64,128]:
    print(f'---------------- {blk_t=} ----------------')
    kernel = inner_chunk_recurrent_fwd_init0(B, t, D, blk_t)
    
    S = torch.empty(B,t // blk_t,D,D).to(k)
    o_tilelang = torch.empty_like(k)
    kernel(S,k,s,o_tilelang)
    if blk_t == 32:
        assert torch.all(o_torch == o_tilelang)
    with torch.profiler.profile() as prof:
        for _ in range(10):
            inner_chunk_recurrent_fwd_init0(B,t,D,blk_t)(S,k,s,o_tilelang)
    print(prof.key_averages().table())