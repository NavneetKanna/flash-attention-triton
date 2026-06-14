import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def self_attn_fwd(
    Q, K, V, O, # (B, H, N, D)
    Cos, Sin, # (N, D) shared across batch/head
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_o_b, stride_o_h, stride_o_n, stride_o_d,
    stride_cos_n, stride_cos_d,
    stride_sin_n, stride_sin_d,
    scale,
    B, H, N, D,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    HALF: tl.constexpr = BLOCK_D // 2
    block_row = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    offset = batch_head_idx * stride_q_h

    # Load the blocks from VRAM to SRAM
    q1_ptr = tl.make_block_ptr(
        base=Q + offset,
        shape=(N, BLOCK_D),
        strides=(stride_q_n, stride_q_d),
        offsets=(block_row * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HALF),
        order=(1, 0) # row major
    )
    q2_ptr = tl.make_block_ptr(
        base=Q + offset,
        shape=(N, BLOCK_D),
        strides=(stride_q_n, stride_q_d),
        offsets=(block_row * BLOCK_Q, HALF),
        block_shape=(BLOCK_Q, HALF),
        order=(1, 0)
    )
    q1 = tl.load(q1_ptr, boundary_check=(0, 1)) # (BLOCK_Q, HALF)
    q2 = tl.load(q2_ptr, boundary_check=(0, 1))

    # cos/sin first half for the Q positions (table halves are identical, so first half suffices)
    cosq_ptr = tl.make_block_ptr(
        base=Cos, shape=(N, BLOCK_D),
        strides=(stride_cos_n, stride_cos_d),
        offsets=(block_row * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HALF),
        order=(1, 0)
    )
    sinq_ptr = tl.make_block_ptr(
        base=Sin, shape=(N, BLOCK_D),
        strides=(stride_sin_n, stride_sin_d),
        offsets=(block_row * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, HALF),
        order=(1, 0)
    )
    cos_q = tl.load(cosq_ptr, boundary_check=(0, 1))
    sin_q = tl.load(sinq_ptr, boundary_check=(0, 1))

    # rotated halves of Q: out1 = x1*cos - x2*sin; out2 = x2*cos + x1*sin
    q_rot1 = q1 * cos_q - q2 * sin_q # (BLOCK_Q, HALF)
    q_rot2 = q2 * cos_q + q1 * sin_q

    mi = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    li = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    q_idx = block_row * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qk_scale = scale * 1.44269504089

    k1_ptr = tl.make_block_ptr(
        base=K + offset,
        shape=(N, BLOCK_D),
        strides=(stride_k_n, stride_k_d),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HALF),
        order=(1, 0) # row major
    )
    k2_ptr = tl.make_block_ptr(
        base=K + offset,
        shape=(N, BLOCK_D),
        strides=(stride_k_n, stride_k_d),
        offsets=(0, HALF),
        block_shape=(BLOCK_KV, HALF),
        order=(1, 0)
    )

    v_block_ptr = tl.make_block_ptr(
        base=V + offset,
        shape=(N, BLOCK_D),
        strides=(stride_v_n, stride_v_d),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, BLOCK_D),
        order=(1, 0)
    )

    cosk_ptr = tl.make_block_ptr(
        base=Cos,
        shape=(N, BLOCK_D),
        strides=(stride_cos_n, stride_cos_d),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HALF),
        order=(1, 0)
    )
    sink_ptr = tl.make_block_ptr(
        base=Sin,
        shape=(N, BLOCK_D),
        strides=(stride_sin_n, stride_sin_d),
        offsets=(0, 0),
        block_shape=(BLOCK_KV, HALF),
        order=(1, 0)
    )

    for start_kv in range(0, N, BLOCK_KV):
        k1 = tl.load(k1_ptr, boundary_check=(0, 1)) # (BLOCK_KV, HALF)
        k2 = tl.load(k2_ptr, boundary_check=(0, 1))
        v_ptr = tl.load(v_block_ptr, boundary_check=(0, 1)) # (BLOCK_KV, BLOCK_D)
        cos_k = tl.load(cosk_ptr, boundary_check=(0, 1))
        sin_k = tl.load(sink_ptr, boundary_check=(0, 1))

        k_idx = start_kv + tl.arange(0, BLOCK_KV)

        # rotated halves of K
        k_rot1 = k1 * cos_k - k2 * sin_k # (BLOCK_KV, HALF)
        k_rot2 = k2 * cos_k + k1 * sin_k

        # q_rot . k_rot over full head-dim == sum over the two halves
        qk = tl.dot(q_rot1, tl.trans(k_rot1)) + tl.dot(q_rot2, tl.trans(k_rot2))
        qk = qk * qk_scale

        # Causal mask
        qk = tl.where(q_idx[:, None] >= k_idx[None, :], qk, float("-inf"))

        # Online softmax
        new_mi = tl.maximum(mi, tl.max(qk, axis=1))
        alpha = tl.math.exp2(mi - new_mi)
        p = tl.math.exp2(qk - new_mi[:, None])

        o_acc = o_acc * alpha[:, None] + tl.dot(p.to(v_ptr.dtype), v_ptr)
        mi = new_mi
        li = li * alpha + tl.sum(p, axis=1)

        k1_ptr = tl.advance(k1_ptr, (BLOCK_KV, 0))
        k2_ptr = tl.advance(k2_ptr, (BLOCK_KV, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_KV, 0))
        cosk_ptr = tl.advance(cosk_ptr, (BLOCK_KV, 0))
        sink_ptr = tl.advance(sink_ptr, (BLOCK_KV, 0))

    o_acc = o_acc / li[:, None]

    o_block_ptr = tl.make_block_ptr(
        base=O + offset,
        shape=(N, BLOCK_D),
        strides=(stride_o_n, stride_o_d),
        offsets=(block_row * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, BLOCK_D),
        order=(1, 0)
    )
    tl.store(o_block_ptr, o_acc.to(O.dtype.element_ty), boundary_check=(0, 1))

def flash_attention(q, k, v, cos, sin):
    B, H, N, D = q.shape
    assert q.shape == k.shape == v.shape
    assert cos.shape == (N, D) and sin.shape == (N, D), "cos/sin must be (N, D)"
    assert D % 2 == 0, "The last dim of Q, K, V needs to be divisble by 2"

    BLOCK_Q = 32
    BLOCK_KV = 32
    assert N % BLOCK_Q == 0 and N % BLOCK_KV == 0
    assert D % 2 == 0

    o = torch.empty_like(q)
    scale = 1.0 / (D ** 0.5)
    grid = (N // BLOCK_Q, B * H)

    self_attn_fwd[grid](
        q, k, v, o,
        cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        scale, B, H, N, D,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, BLOCK_D=D,
    )
    return o

def precompute_rope(D, N, base=10000.0, device="cuda", dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=device).float() / D))
    t = torch.arange(N, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype) # each (N, D)

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def apply_rope(x, cos, sin): # x:(B,H,N,D), cos/sin:(N,D)
    return x * cos + rotate_half(x) * sin

if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU + Triton"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = parser.parse_args()
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    atol = {"fp16": 2e-2, "bf16": 3e-2, "fp32": 1e-2}[args.dtype]

    torch.manual_seed(0)
    B, H, N, D = 2, 4, 256, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=dt)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dt)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dt)

    cos, sin = precompute_rope(D, N, device="cuda", dtype=dt)

    out = flash_attention(q, k, v, cos, sin)

    qr = apply_rope(q, cos, sin)
    kr = apply_rope(k, cos, sin)
    ref = F.scaled_dot_product_attention(qr, kr, v, is_causal=True)

    torch.testing.assert_close(out, ref, atol=atol, rtol=0)
    print(f"[{args.dtype}] passed (max diff {(out - ref).abs().max().item():.2e})")

"""

Now let's understand the RoPE part using the same example.

Same setup as before:
 BLOCK_Q = 4, BLOCK_KV = 4, BLOCK_D = 4 -> HALF = BLOCK_D // 2 = 2
 Q, K, V shape (2, 2, 8, 4), and for block 0 we take head (0, 0) with Q = K = V.

block 0:
 block_row = 0
 batch_head_idx = 0
 offset = 0

The non-RoPE machinery (online softmax + dot with V) is exactly as explained
in the other file. The only new thing is what happens to Q and K BEFORE the dot product.
Note: Cos/Sin are shape (N, D) and shared across batch/head, so they are indexed
by position only, we never add offset to them.

STEP 1: load Q as TWO halves (we can't slice a loaded tile in triton, so we
        make two block pointers into the same rows, cols 0..1 and cols 2..3).

q1 (cols 0-1)              q2 (cols 2-3)
    [0.3581, 0.1616]           [0.5714, 0.4795]
    [0.5468, 0.3008]           [0.9154, 0.3457]
    [0.4201, 0.1406]           [0.2273, 0.5269]
    [0.1441, 0.1024]           [0.8580, 0.8310]

STEP 2: load cos/sin (FIRST HALF only) for positions 0..3.

With D = 4 the two pair-speeds are [1.0, 0.01], so position p has angles
[p*1.0, p*0.01]. We only load the first half because the table's two halves
are identical (emb = cat([angles, angles])).

cos_q                      sin_q
    [ 1.0000,  1.0000]         [0.0000, 0.0000] <- pos 0
    [ 0.5403,  0.9999]         [0.8415, 0.0100] <- pos 1
    [-0.4161,  0.9998]         [0.9093, 0.0200] <- pos 2
    [-0.9900,  0.9996]         [0.1411, 0.0300] <- pos 3

STEP 3: rotate, using arithmetic on the halves.
        q_rot1 = q1*cos_q - q2*sin_q
        q_rot2 = q2*cos_q + q1*sin_q

q_rot1                     q_rot2
    [ 0.3581,  0.1616]         [ 0.5714,  0.4795] <- pos 0 unchanged (cos=1, sin=0)
    [-0.4748,  0.2973]         [ 0.9547,  0.3487] <- pos 1 turned a bit
    [-0.3815,  0.1300]         [ 0.2874,  0.5296] <- pos 2 turned more
    [-0.2637,  0.0774]         [-0.8291,  0.8337] <- pos 3 turned most

Position 0 comes out identical to the input (no rotation). The further down
the block, the more the values swing, that is the position being encoded.

STEP 4: K gets the exact same treatment. In this example, Q = K and the KV block also
        covers positions 0..3, so k_rot1 = q_rot1 and k_rot2 = q_rot2.

STEP 5: the QK dot, done as TWO half matmuls and summed.
        Instead of stitching the halves back into one (4,4) tile, we keep them
        split and add the two half dots. This is just the distributive property
        of the dot product: q_rot . k_rot = (over first half) + (over second half).

part1 = q_rot1 @ k_rot1.T             part2 = q_rot2 @ k_rot2.T
    [ 0.1544 -0.1220 -0.1156 -0.0819]   [ 0.5564  0.7127  0.4182 -0.0740]
    [-0.1220  0.3139  0.2198  0.1483]   [ 0.7127  1.0331  0.4591 -0.5008]
    [-0.1156  0.2198  0.1625  0.1107]   [ 0.4182  0.4591  0.3631  0.2032]
    [-0.0819  0.1483  0.1107  0.0756]   [-0.0740 -0.5008  0.2032  1.3824]

qk = part1 + part2
    [ 0.7108  0.5907  0.3026 -0.1559]
    [ 0.5907  1.3469  0.6789 -0.3526]
    [ 0.3026  0.6789  0.5255  0.3139]
    [-0.1559 -0.3526  0.3139  1.4580]

STEP 6: from here the original kernel takes over unchanged. Causal mask keeps
        the lower triangle (q_idx >= k_idx):

    [ 0.7108    -inf    -inf    -inf  ]
    [ 0.5907  1.3469    -inf    -inf  ]
    [ 0.3026  0.6789  0.5255    -inf  ]
    [-0.1559 -0.3526  0.3139  1.4580  ]

then online softmax over this, and dot with V. V is never rotated.

So to summarize the RoPE additions to block 0:
 - Load Q (and K) as two halves via separate block pointers (cols 0..HALF and HALF..D),
   instead of one (BLOCK_Q, BLOCK_D) tile, triton can't slice a loaded tile.
 - Load the first half of cos/sin for the block's positions (table halves are identical).
   Index by position only; no offset needed (Cos/Sin have no batch/head dim).
 - Rotate with two lines: q_rot1 = q1*cos - q2*sin; q_rot2 = q2*cos + q1*sin.
 - Compute qk = q_rot1 @ k_rot1.T + q_rot2 @ k_rot2.T (two half dots, summed).
 - Everything after (causal mask, online softmax, @ V) is the original kernel.

"""
