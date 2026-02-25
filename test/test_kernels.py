"""
Correctness tests for common GPU kernels on the tileon interpreter.
Covers: GEMM, Softmax, FlashAttention-2, Sparse (Block-Sparse) Attention.
"""
import math
import numpy as np
import os
import torch

import tileon
import tileon.knobs
import tileon.language as tl
import tileon.language.random as tl_random

DEVICE = torch.device("cpu")
torch.manual_seed(42)


# ===================== GEMM =====================

@tileon.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


def torch_matmul(a, b):
    return a @ b


def numpy_matmul(a, b):
    return np.matmul(a, b)


def test_gemm(M, N, K):
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
    c = torch.empty(M, N, device=DEVICE, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 16
    if M <= 16 and N <= 16:
        BLOCK_M, BLOCK_N = 16, 16
    if K <= 16:
        BLOCK_K = 16

    grid = (tileon.cdiv(M, BLOCK_M), tileon.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    ref = torch_matmul(a, b)
    assert torch.allclose(c, ref, atol=1e-2, rtol=1e-2), \
        f"GEMM mismatch: max diff={torch.max(torch.abs(c - ref)).item()}"


# ===================== Softmax =====================

@tileon.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_ptr = input_ptr + row_idx * stride_row + col_offsets
    row = tl.load(row_ptr, mask=mask, other=float('-inf'))

    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    out_ptr = output_ptr + row_idx * stride_row + col_offsets
    tl.store(out_ptr, result, mask=mask)


def test_softmax(rows, cols):
    x = torch.randn(rows, cols, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)

    BLOCK_SIZE = max(32, tileon.next_power_of_2(cols))
    softmax_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_SIZE=BLOCK_SIZE)

    ref = torch.softmax(x, dim=-1)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), \
        f"Softmax mismatch: max diff={torch.max(torch.abs(out - ref)).item()}"


def numpy_softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ===================== FlashAttention-2 (forward) =====================

@tileon.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    sm_scale,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Load Q block
    q_ptrs = Q_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = (offs_m[:, None] < N_CTX)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Iterate over K/V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block
        k_ptrs = K_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = (offs_n[:, None] < N_CTX)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # S = Q @ K^T * scale
        s = tl.dot(q, tl.transpose(k)) * sm_scale
        # Mask out-of-bounds
        s_mask = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        s = tl.where(s_mask, s, float('-inf'))

        # Online softmax update
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Fix RuntimeWarning: invalid value encountered in subtract
        # When m_new is -inf (padding rows), m_i is also -inf, leading to NaN.
        # We replace -inf with 0.0 to safely compute alpha=1.0 and exp(s-m_new)=0.0
        m_new_is_inf = m_new == float('-inf')
        m_i_safe = tl.where(m_new_is_inf, 0.0, m_i)
        m_new_safe = tl.where(m_new_is_inf, 0.0, m_new)

        alpha = tl.exp(m_i_safe - m_new_safe)
        l_i = l_i * alpha + tl.sum(tl.exp(s - m_new_safe[:, None]), axis=1)

        # Update accumulator
        acc = acc * alpha[:, None]
        # Load V block
        v_ptrs = V_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = (offs_n[:, None] < N_CTX)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        p = tl.exp(s - m_new_safe[:, None])
        acc += tl.dot(p.to(tl.float32), v)

        m_i = m_new

    # Normalize
    # Avoid division by zero for padding rows where l_i is 0
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    # Store output
    o_ptrs = O_ptr + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = (offs_m[:, None] < N_CTX)
    tl.store(o_ptrs, acc, mask=o_mask)


def torch_flash_attention(Q, K, V, sm_scale):
    S = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


def numpy_flash_attention(Q, K, V, sm_scale):
    S = np.matmul(Q, np.transpose(K, (0, 1, 3, 2))) * sm_scale
    P = softmax_numpy_4d(S, axis=-1)
    return np.matmul(P, V)


def softmax_numpy_4d(x, axis):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def test_flash_attention_fwd(B, H, N, D):
    Q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    K = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    V = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float32)
    O = torch.empty_like(Q)

    sm_scale = 1.0 / math.sqrt(D)
    BLOCK_M = min(32, N)
    BLOCK_N = min(32, N)

    grid = (tileon.cdiv(N, BLOCK_M), B * H)
    flash_attention_fwd_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        sm_scale=sm_scale,
    )

    ref = torch_flash_attention(Q, K, V, sm_scale)
    assert torch.allclose(O, ref, atol=1e-2, rtol=1e-2), \
        f"FA2 mismatch: max diff={torch.max(torch.abs(O - ref)).item()}"


# ===================== Random =====================

@tileon.jit
def random_kernel(
    output_ptr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    seed_tensor = tl.full(offsets.shape, seed, dtype=tl.int64)
    rand_vals = tl_random.rand(seed_tensor, offsets)
    tl.store(output_ptr + offsets, rand_vals, mask=mask)


@tileon.jit
def randn_kernel(
    output_ptr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    seed_tensor = tl.full(offsets.shape, seed, dtype=tl.int64)
    randn_vals = tl_random.randn(seed_tensor, offsets)
    tl.store(output_ptr + offsets, randn_vals, mask=mask)


@tileon.jit
def randint_kernel(
    output_ptr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    seed_tensor = tl.full(offsets.shape, seed, dtype=tl.int64)
    randint_vals = tl_random.randint(seed_tensor, offsets)
    tl.store(output_ptr + offsets, randint_vals, mask=mask)


def test_random(size):
    output = torch.empty(size, device=DEVICE, dtype=torch.float32)
    seed = 42
    BLOCK_SIZE = 128
    grid = lambda meta: (tileon.cdiv(size, meta['BLOCK_SIZE']), )
    random_kernel[grid](output, size, seed=seed, BLOCK_SIZE=BLOCK_SIZE)
    assert output.min() >= 0.0 and output.max() < 1.0, \
        f"rand not in [0, 1): min={output.min()}, max={output.max()}"


def test_randn(size):
    output = torch.empty(size, device=DEVICE, dtype=torch.float32)
    seed = 42
    BLOCK_SIZE = 128
    grid = lambda meta: (tileon.cdiv(size, meta['BLOCK_SIZE']), )
    randn_kernel[grid](output, size, seed=seed, BLOCK_SIZE=BLOCK_SIZE)
    mean = output.mean().item()
    std = output.std().item()
    assert abs(mean) < 0.1 and abs(std - 1.0) < 0.2, \
        f"randn not ~N(0,1): mean={mean}, std={std}"


def test_randint(size):
    output = torch.empty(size, device=DEVICE, dtype=torch.int32)
    seed = 42
    BLOCK_SIZE = 128
    grid = lambda meta: (tileon.cdiv(size, meta['BLOCK_SIZE']), )
    randint_kernel[grid](output, size, seed=seed, BLOCK_SIZE=BLOCK_SIZE)
    assert output.min() >= torch.iinfo(torch.int32).min and output.max() <= torch.iinfo(torch.int32).max, \
        f"randint out of int32 range"


# ===================== Block-Sparse Attention =====================

@tileon.jit
def block_sparse_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    block_indices_ptr,  # (num_q_blocks, max_nnz) int32 indices of which K/V blocks to attend to
    num_blocks_per_row_ptr,  # (num_q_blocks,) int32 actual nnz per row
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    stride_om, stride_ok,
    stride_idx_row, stride_idx_col,
    N_CTX, max_nnz,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    sm_scale,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Load Q block
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # How many K/V blocks this Q block attends to
    nnz = tl.load(num_blocks_per_row_ptr + pid_m)

    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for idx in range(max_nnz):
        # Check if this index is valid
        if idx < nnz:
            # Load which K/V block to attend to
            block_idx = tl.load(block_indices_ptr + pid_m * stride_idx_row + idx * stride_idx_col)
            offs_n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

            # Load K block
            k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k_mask = offs_n[:, None] < N_CTX
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # S = Q @ K^T * scale
            s = tl.dot(q, tl.transpose(k)) * sm_scale
            s_mask = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
            s = tl.where(s_mask, s, float('-inf'))

            # Online softmax
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            # Fix RuntimeWarning: invalid value encountered in subtract
            m_new_is_inf = m_new == float('-inf')
            m_i_safe = tl.where(m_new_is_inf, 0.0, m_i)
            m_new_safe = tl.where(m_new_is_inf, 0.0, m_new)

            alpha = tl.exp(m_i_safe - m_new_safe)
            l_i = l_i * alpha + tl.sum(tl.exp(s - m_new_safe[:, None]), axis=1)
            acc = acc * alpha[:, None]

            # Load V and accumulate
            v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v_mask = offs_n[:, None] < N_CTX
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            p = tl.exp(s - m_new_safe[:, None])
            acc += tl.dot(p.to(tl.float32), v)

            m_i = m_new

    # Avoid division by zero for padding rows where l_i is 0
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]

    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, acc, mask=o_mask)


def torch_block_sparse_attention(Q, K, V, block_indices, num_blocks_per_row, BLOCK_M, BLOCK_N, sm_scale):
    """Reference: dense attention masked by block sparsity pattern."""
    N, D = Q.shape
    num_q_blocks = tileon.cdiv(N, BLOCK_M)
    O = torch.zeros_like(Q)

    for qb in range(num_q_blocks):
        q_start = qb * BLOCK_M
        q_end = min(q_start + BLOCK_M, N)
        q_block = Q[q_start:q_end]

        nnz = num_blocks_per_row[qb].item()
        if nnz == 0:
            continue

        kv_indices = []
        for j in range(nnz):
            kb = block_indices[qb, j].item()
            k_start = kb * BLOCK_N
            k_end = min(k_start + BLOCK_N, N)
            kv_indices.extend(range(k_start, k_end))

        k_gathered = K[kv_indices]
        v_gathered = V[kv_indices]

        s = (q_block @ k_gathered.T) * sm_scale
        p = torch.softmax(s, dim=-1)
        O[q_start:q_end] = p @ v_gathered

    return O


def numpy_block_sparse_attention(Q, K, V, block_indices, num_blocks_per_row, BLOCK_M, BLOCK_N, sm_scale):
    """Reference: dense attention masked by block sparsity pattern."""
    N, D = Q.shape
    num_q_blocks = int(np.ceil(N / BLOCK_M))
    O = np.zeros((N, D), dtype=np.float32)

    for qb in range(num_q_blocks):
        q_start = qb * BLOCK_M
        q_end = min(q_start + BLOCK_M, N)
        q_block = Q[q_start:q_end]

        nnz = num_blocks_per_row[qb]
        if nnz == 0:
            continue

        kv_indices = []
        for j in range(nnz):
            kb = block_indices[qb, j]
            k_start = kb * BLOCK_N
            k_end = min(k_start + BLOCK_N, N)
            kv_indices.extend(range(k_start, k_end))

        k_gathered = K[kv_indices]
        v_gathered = V[kv_indices]

        s = (q_block @ k_gathered.T) * sm_scale
        s_max = np.max(s, axis=-1, keepdims=True)
        exp_s = np.exp(s - s_max)
        p = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        O[q_start:q_end] = p @ v_gathered

    return O

def test_block_sparse_attention(N, D, sparsity):
    BLOCK_M = 32
    BLOCK_N = 32
    num_q_blocks = tileon.cdiv(N, BLOCK_M)
    num_k_blocks = tileon.cdiv(N, BLOCK_N)
    sm_scale = 1.0 / math.sqrt(D)

    Q = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
    K = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
    V = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
    O = torch.empty_like(Q)

    # Generate block sparsity pattern: each Q block attends to a random subset of K blocks
    # Ensure at least 1 block per row
    max_nnz = max(1, int(num_k_blocks * (1 - sparsity)))
    block_indices = torch.zeros(num_q_blocks, max_nnz, dtype=torch.int32, device=DEVICE)
    num_blocks_per_row = torch.zeros(num_q_blocks, dtype=torch.int32, device=DEVICE)

    for qb in range(num_q_blocks):
        nnz = max_nnz
        chosen = torch.randperm(num_k_blocks)[:nnz].sort().values
        block_indices[qb, :nnz] = chosen.to(torch.int32)
        num_blocks_per_row[qb] = nnz

    grid = (num_q_blocks,)
    block_sparse_attn_kernel[grid](
        Q, K, V, O,
        block_indices, num_blocks_per_row,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        block_indices.stride(0), block_indices.stride(1),
        N, max_nnz,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        sm_scale=sm_scale,
    )

    ref = torch_block_sparse_attention(Q, K, V, block_indices, num_blocks_per_row, BLOCK_M, BLOCK_N, sm_scale)
    assert torch.allclose(O, ref, atol=1e-2, rtol=1e-2), \
        f"Block-sparse attention mismatch: max diff={torch.max(torch.abs(O - ref)).item()}"


@tileon.testing.perf_report(
    tileon.testing.Benchmark(
        x_names=['size'],
        x_vals=[64, 128, 256] if tileon.knobs.runtime.interpret else [64, 128, 256, 512],
        x_log=True,
        line_arg='provider',
        line_vals=['tileon', 'torch', 'numpy'],
        line_names=['Tileon', 'Torch', 'NumPy'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GFLOPS',
        plot_name='gemm-performance',
        args={},
    )
)
def benchmark_gemm(size, provider):
    M = N = K = size
    a = torch.rand(M, K, device=DEVICE, dtype=torch.float32)
    b = torch.rand(K, N, device=DEVICE, dtype=torch.float32)
    c = torch.empty(M, N, device=DEVICE, dtype=torch.float32)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 16

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: torch_matmul(a, b), quantiles=quantiles)
    if provider == 'tileon':
        grid = (tileon.cdiv(M, BLOCK_M), tileon.cdiv(N, BLOCK_N))
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ), quantiles=quantiles)
    if provider == 'numpy':
        a_np = a.cpu().numpy()
        b_np = b.cpu().numpy()
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: numpy_matmul(a_np, b_np), quantiles=quantiles)

    flops = 2 * M * N * K
    gflops = lambda ms: flops * 1e-9 / (ms * 1e-3)
    return gflops(ms), gflops(max_ms), gflops(min_ms)


@tileon.testing.perf_report(
    tileon.testing.Benchmark(
        x_names=['cols'],
        x_vals=[64, 128, 256] if tileon.knobs.runtime.interpret else [64, 128, 256, 512],
        x_log=True,
        line_arg='provider',
        line_vals=['tileon', 'torch', 'numpy'],
        line_names=['Tileon', 'Torch', 'NumPy'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='softmax-performance',
        args={},
    )
)
def benchmark_softmax(cols, provider):
    rows = 8
    x = torch.rand(rows, cols, device=DEVICE, dtype=torch.float32)
    out = torch.empty_like(x)
    BLOCK_SIZE = max(32, tileon.next_power_of_2(cols))

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: torch.softmax(x, dim=-1), quantiles=quantiles)
    if provider == 'tileon':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: softmax_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK_SIZE=BLOCK_SIZE), quantiles=quantiles)
    if provider == 'numpy':
        x_np = x.cpu().numpy()
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: numpy_softmax(x_np), quantiles=quantiles)

    gbps = lambda ms: rows * cols * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@tileon.testing.perf_report(
    tileon.testing.Benchmark(
        x_names=['size'],
        x_vals=[64, 128, 256] if tileon.knobs.runtime.interpret else [64, 128, 256, 512],
        x_log=True,
        line_arg='provider',
        line_vals=['tileon', 'torch', 'numpy'],
        line_names=['Tileon', 'Torch', 'NumPy'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='flash-attention-performance',
        args={},
    )
)
def benchmark_flash_attention(size, provider):
    N = size
    B = 1
    H = 1
    D = 32
    Q = torch.rand(B, H, N, D, device=DEVICE, dtype=torch.float32)
    K = torch.rand(B, H, N, D, device=DEVICE, dtype=torch.float32)
    V = torch.rand(B, H, N, D, device=DEVICE, dtype=torch.float32)
    O = torch.empty_like(Q)

    sm_scale = 1.0 / math.sqrt(D)
    BLOCK_M = min(32, N)
    BLOCK_N = min(32, N)

    grid = (tileon.cdiv(N, BLOCK_M), B * H)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: torch_flash_attention(Q, K, V, sm_scale), quantiles=quantiles)
    if provider == 'tileon':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: flash_attention_fwd_kernel[grid](
            Q, K, V, O,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
            sm_scale=sm_scale,
        ), quantiles=quantiles)
    if provider == 'numpy':
        Q_np = Q.cpu().numpy()
        K_np = K.cpu().numpy()
        V_np = V.cpu().numpy()
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: numpy_flash_attention(Q_np, K_np, V_np, sm_scale), quantiles=quantiles)

    gbps = lambda ms: 4 * B * H * N * N * D * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@tileon.testing.perf_report(
    tileon.testing.Benchmark(
        x_names=['size'],
        x_vals=[64, 128, 256] if tileon.knobs.runtime.interpret else [64, 128, 256, 512],
        x_log=True,
        line_arg='provider',
        line_vals=['tileon', 'torch', 'numpy'],
        line_names=['Tileon', 'Torch', 'NumPy'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='block-sparse-attention-performance',
        args={},
    )
)
def benchmark_block_sparse_attention(size, provider):
    N = size
    D = 32
    BLOCK_M = 32
    BLOCK_N = 32
    num_q_blocks = tileon.cdiv(N, BLOCK_M)
    num_k_blocks = tileon.cdiv(N, BLOCK_N)
    sm_scale = 1.0 / math.sqrt(D)
    sparsity = 0.5

    Q = torch.rand(N, D, device=DEVICE, dtype=torch.float32)
    K = torch.rand(N, D, device=DEVICE, dtype=torch.float32)
    V = torch.rand(N, D, device=DEVICE, dtype=torch.float32)
    O = torch.empty_like(Q)

    max_nnz = max(1, int(num_k_blocks * (1 - sparsity)))
    block_indices = torch.zeros(num_q_blocks, max_nnz, dtype=torch.int32, device=DEVICE)
    num_blocks_per_row = torch.zeros(num_q_blocks, dtype=torch.int32, device=DEVICE)

    for qb in range(num_q_blocks):
        nnz = max_nnz
        chosen = torch.randperm(num_k_blocks)[:nnz].sort().values
        block_indices[qb, :nnz] = chosen.to(torch.int32)
        num_blocks_per_row[qb] = nnz

    grid = (num_q_blocks,)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = tileon.testing.do_bench(
            lambda: torch_block_sparse_attention(Q, K, V, block_indices, num_blocks_per_row, BLOCK_M, BLOCK_N, sm_scale),
            quantiles=quantiles
        )
    if provider == 'tileon':
        ms, min_ms, max_ms = tileon.testing.do_bench(
            lambda: block_sparse_attn_kernel[grid](
                Q, K, V, O,
                block_indices, num_blocks_per_row,
                Q.stride(0), Q.stride(1),
                K.stride(0), K.stride(1),
                V.stride(0), V.stride(1),
                O.stride(0), O.stride(1),
                block_indices.stride(0), block_indices.stride(1),
                N, max_nnz,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
                sm_scale=sm_scale,
            ),
            quantiles=quantiles
        )
    if provider == 'numpy':
        Q_np = Q.cpu().numpy()
        K_np = K.cpu().numpy()
        V_np = V.cpu().numpy()
        block_indices_np = block_indices.cpu().numpy()
        num_blocks_per_row_np = num_blocks_per_row.cpu().numpy()
        ms, min_ms, max_ms = tileon.testing.do_bench(
            lambda: numpy_block_sparse_attention(Q_np, K_np, V_np, block_indices_np, num_blocks_per_row_np, BLOCK_M, BLOCK_N, sm_scale),
            quantiles=quantiles
        )

    gbps = lambda ms: 4 * N * N * D * (1 - sparsity) * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@tileon.testing.perf_report(
    tileon.testing.Benchmark(
        x_names=['size'],
        x_vals=[1024, 2048, 4096] if tileon.knobs.runtime.interpret else [1024, 2048, 4096, 8192],
        x_log=True,
        line_arg='provider',
        line_vals=['tileon', 'torch', 'numpy'],
        line_names=['Tileon', 'Torch', 'NumPy'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='rand-performance',
        args={},
    )
)
def benchmark_random(size, provider):
    output = torch.empty(size, device=DEVICE, dtype=torch.float32)
    seed = 42
    BLOCK_SIZE = 128
    grid = lambda meta: (tileon.cdiv(size, meta['BLOCK_SIZE']), )

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: torch.rand(size, device=DEVICE), quantiles=quantiles)
    if provider == 'tileon':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: random_kernel[grid](output, size, seed=seed, BLOCK_SIZE=BLOCK_SIZE), quantiles=quantiles)
    if provider == 'numpy':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: np.random.rand(size), quantiles=quantiles)

    gbps = lambda ms: size * output.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@tileon.testing.perf_report(
    tileon.testing.Benchmark(
        x_names=['size'],
        x_vals=[1024, 2048, 4096] if tileon.knobs.runtime.interpret else [1024, 2048, 4096, 8192],
        x_log=True,
        line_arg='provider',
        line_vals=['tileon', 'torch', 'numpy'],
        line_names=['Tileon', 'Torch', 'NumPy'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='randn-performance',
        args={},
    )
)
def benchmark_randn(size, provider):
    output = torch.empty(size, device=DEVICE, dtype=torch.float32)
    seed = 42
    BLOCK_SIZE = 128
    grid = lambda meta: (tileon.cdiv(size, meta['BLOCK_SIZE']), )

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: torch.randn(size, device=DEVICE), quantiles=quantiles)
    if provider == 'tileon':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: randn_kernel[grid](output, size, seed=seed, BLOCK_SIZE=BLOCK_SIZE), quantiles=quantiles)
    if provider == 'numpy':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: np.random.randn(size), quantiles=quantiles)

    gbps = lambda ms: size * output.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@tileon.testing.perf_report(
    tileon.testing.Benchmark(
        x_names=['size'],
        x_vals=[1024, 2048, 4096] if tileon.knobs.runtime.interpret else [1024, 2048, 4096, 8192],
        x_log=True,
        line_arg='provider',
        line_vals=['tileon', 'torch', 'numpy'],
        line_names=['Tileon', 'Torch', 'NumPy'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='randint-performance',
        args={},
    )
)
def benchmark_randint(size, provider):
    output = torch.empty(size, device=DEVICE, dtype=torch.int32)
    seed = 42
    BLOCK_SIZE = 128
    grid = lambda meta: (tileon.cdiv(size, meta['BLOCK_SIZE']), )

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: torch.randint(0, 2**31, (size,), device=DEVICE, dtype=torch.int32), quantiles=quantiles)
    if provider == 'tileon':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: randint_kernel[grid](output, size, seed=seed, BLOCK_SIZE=BLOCK_SIZE), quantiles=quantiles)
    if provider == 'numpy':
        ms, min_ms, max_ms = tileon.testing.do_bench(lambda: np.random.randint(0, 2**31, size, dtype=np.int32), quantiles=quantiles)

    gbps = lambda ms: size * 4 * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    benchmark_gemm.run(print_data=True, save_path=os.path.join(os.path.dirname(__file__), 'figures', 'gemm'))
    benchmark_softmax.run(print_data=True, save_path=os.path.join(os.path.dirname(__file__), 'figures', 'softmax'))
    benchmark_flash_attention.run(print_data=True, save_path=os.path.join(os.path.dirname(__file__), 'figures', 'flash-attention'))
    benchmark_block_sparse_attention.run(print_data=True, save_path=os.path.join(os.path.dirname(__file__), 'figures', 'block-sparse-attention'))
    benchmark_random.run(print_data=True, save_path=os.path.join(os.path.dirname(__file__), 'figures', 'random'))
    benchmark_randn.run(print_data=True, save_path=os.path.join(os.path.dirname(__file__), 'figures', 'randn'))
    benchmark_randint.run(print_data=True, save_path=os.path.join(os.path.dirname(__file__), 'figures', 'randint'))
