import argparse
import torch
import torch.nn.functional as F
import triton

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fa_base import flash_attention as fa_triton
from fa_rope import (
    flash_attention as fa_triton_rope,
    precompute_rope,
    apply_rope,
)

try:
    from flash_attn import flash_attn_func
    HAS_FA = True
except Exception:
    HAS_FA = False

import os
os.makedirs("assets", exist_ok=True)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
ATOL = {"fp16": 2e-2, "bf16": 3e-2, "fp32": 1e-2}

def attn_flops(B, H, N, D, causal=True):
    # QK.T and PV are each 2*B*H*N*N*D (multiply + add). Causal ~ half the work.
    f = 4.0 * B * H * N * N * D
    return f * 0.5 if causal else f

def bench(fn, flops):
    """Return (median_ms, q20_ms, q80_ms, tflops_at_median)."""
    med, lo, hi = triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
    tflops = flops / (med * 1e-3) / 1e12
    return med, lo, hi, tflops

def make_inputs(B, H, N, D, dtype, device="cuda"):
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    cos, sin = precompute_rope(D, N, device=device, dtype=dtype)
    return q, k, v, cos, sin

def correct(name, out, ref, atol):
    try:
        torch.testing.assert_close(out, ref, atol=atol, rtol=0)
        return True
    except AssertionError as e:
        md = (out - ref).abs().max().item()
        print(f"[WARN] {name} failed correctness (max diff {md:.2e}); timing anyway")
        return False

def run(args):
    dtype = DTYPES[args.dtype]
    atol = ATOL[args.dtype]
    B, H, D = args.batch, args.heads, args.head_dim
    seqlens = args.seqlens

    print(f"# config: B={B} H={H} D={D} dtype={args.dtype} causal=True  (forward only)\n")

    # group 1: no-RoPE attention
    rows_rt = [] # runtime rows
    rows_tf = [] # tflops rows
    for N in seqlens:
        q, k, v, cos, sin = make_inputs(B, H, N, D, dtype)
        flops = attn_flops(B, H, N, D, causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        results = {}

        # torch naive
        def naive():
            s = (q @ k.transpose(-2, -1)) * (1.0 / (D ** 0.5))
            mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), 1)
            s = s.masked_fill(mask, float("-inf"))
            return torch.softmax(s, dim=-1) @ v

        correct("naive", naive(), ref, atol)
        results["naive"] = bench(naive, flops)

        # torch SDPA
        results["sdpa"] = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), flops)

        # our triton kernel
        correct("triton", fa_triton(q, k, v), ref, atol)
        results["triton"] = bench(lambda: fa_triton(q, k, v), flops)

        # official flash-attn (expects (B, N, H, D) layout)
        if HAS_FA:
            qf, kf, vf = (t.transpose(1, 2).contiguous() for t in (q, k, v))
            results["flash_attn"] = bench(
                lambda: flash_attn_func(qf, kf, vf, causal=True), flops)

        rows_rt.append((N, {k2: (r[0] if r else None) for k2, r in results.items()}))
        rows_tf.append((N, {k2: (r[3] if r else None) for k2, r in results.items()}))

    # group 2: RoPE
    print("\n# RoPE group (fused vs rotate-outside)\n")
    rope_rows = []
    for N in seqlens:
        q, k, v, cos, sin = make_inputs(B, H, N, D, dtype)
        flops = attn_flops(B, H, N, D, causal=True)

        # reference for this group: rotate then SDPA
        qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        ref = F.scaled_dot_product_attention(qr, kr, v, is_causal=True)

        def outside_sdpa():
            qr = apply_rope(q, cos, sin); kr = apply_rope(k, cos, sin)
            return F.scaled_dot_product_attention(qr, kr, v, is_causal=True)

        def outside_triton():
            qr = apply_rope(q, cos, sin); kr = apply_rope(k, cos, sin)
            return fa_triton(qr, kr, v)

        def fused():
            return fa_triton_rope(q, k, v, cos, sin)

        correct("fused-rope", fused(), ref, atol)
        r_out_sdpa = bench(outside_sdpa, flops)
        r_out_tri = bench(outside_triton, flops)
        r_fused = bench(fused, flops)
        rope_rows.append((N, r_out_sdpa[0], r_out_tri[0], r_fused[0]))

    # markdown tables
    print("\n\n## runtime (ms), no-RoPE\n")
    cols = list(rows_rt[0][1].keys())
    print("| seq_len | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for N, d in rows_rt:
        print(f"| {N} | " + " | ".join(
            (f"{d[c]:.3f}" if d.get(c) is not None else "OOM") for c in cols) + " |")

    print("\n## achieved TFLOP/s, no-RoPE\n")
    print("| seq_len | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for N, d in rows_tf:
        print(f"| {N} | " + " | ".join(
            (f"{d[c]:.1f}" if d.get(c) is not None else "-") for c in cols) + " |")

    print("\n## RoPE fusion (ms)\n")
    print("| seq_len | rotate-outside + SDPA | rotate-outside + our kernel | fused |")
    print("|---|---|---|---|")
    for N, a, b, c in rope_rows:
        print(f"| {N} | {a:.3f} | {b:.3f} | {c:.3f} |")

    # plots
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in cols:
        ys = [d.get(c) for _, d in rows_rt]
        xs = [N for N, _ in rows_rt]
        xs2 = [x for x, y in zip(xs, ys) if y is not None]
        ys2 = [y for y in ys if y is not None]
        ax.plot(xs2, ys2, marker="o", label=c)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("sequence length"); ax.set_ylabel("runtime (ms)")
    ax.set_title(f"Causal attention forward, {args.dtype}")
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig("assets/runtime_vs_seqlen.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in cols:
        ys = [d.get(c) for _, d in rows_tf]
        xs = [N for N, _ in rows_tf]
        xs2 = [x for x, y in zip(xs, ys) if y is not None]
        ys2 = [y for y in ys if y is not None]
        ax.plot(xs2, ys2, marker="o", label=c)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"Causal attention forward, {args.dtype}")
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig("assets/tflops_vs_seqlen.png", dpi=150)

    # RoPE runtime plot. The clean fusion comparison holds the attention
    # engine fixed (our kernel) and varies only WHERE RoPE happens, so the
    # two solid lines are fused vs rotate-outside+our-kernel. SDPA is drawn
    # faint as a "fastest available" reference, it swaps in a
    # different attention engine (FA2), so it's a reference, not the bar
    # we draw the fusion conclusion from.
    xs = [N for N, *_ in rope_rows]
    out_sdpa = [r[1] for r in rope_rows]
    out_tri = [r[2] for r in rope_rows]
    fused = [r[3] for r in rope_rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, fused, marker="o", label="fused (our kernel)")
    ax.plot(xs, out_tri, marker="o", label="rotate-outside + our kernel")
    ax.plot(xs, out_sdpa, marker="o", alpha=0.45,
            label="rotate-outside + SDPA (FA2 ref.)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("runtime (ms)")
    ax.set_title(f"RoPE + causal attention forward, {args.dtype}")
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig("assets/rope_runtime_vs_seqlen.png", dpi=150)

    print("\nsaved runtime_vs_seqlen.png, tflops_vs_seqlen.png, rope_runtime_vs_seqlen.png in assets/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--seqlens", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096, 8192])
    args = p.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    run(args)

