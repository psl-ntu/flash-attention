#!/usr/bin/env python3
"""
FA3 benchmark — Llama 3 attention dimensions.

Modes:
  prefill  Sweeps seqlen (seqlen_q = seqlen_k), fp16/bf16/fp8, causal/non-causal, fwd/bwd.
  decode   seqlen_q=1, sweeps KV-cache length, fp16/bf16/fp8, multiple batch sizes, fwd only.

Compares Flash Attention 3 against PyTorch SDPA (flash-attention backend).

Usage:
    python benchmarks/benchmark_fa3.py
    python benchmarks/benchmark_fa3.py --model llama3-70b
    python benchmarks/benchmark_fa3.py --no-bwd --no-decode
    python benchmarks/benchmark_fa3.py --decode-batches 1 8 32 64
    python benchmarks/benchmark_fa3.py --seqlens 1024 4096 16384 --no-fp8
"""
import argparse

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from flash_attn_3.flash_attn_interface import flash_attn_func as _fa3_func
from flash_attn_3.flash_attn_interface import flash_attn_with_kvcache as _fa3_kvcache_func

# ── Model configs ─────────────────────────────────────────────────────────────
_MODELS = {
    "llama3-8b":  dict(label="Llama-3 8B",  nheads=32, nheads_kv=8,  headdim=128),
    "llama3-70b": dict(label="Llama-3 70B", nheads=64, nheads_kv=8,  headdim=128),
}
DEFAULT_MODEL = "llama3-8b"

# Globals filled in main() after --model is parsed
MODEL      = _MODELS[DEFAULT_MODEL]["label"]
NHEADS     = _MODELS[DEFAULT_MODEL]["nheads"]
NHEADS_KV  = _MODELS[DEFAULT_MODEL]["nheads_kv"]
HEADDIM    = _MODELS[DEFAULT_MODEL]["headdim"]
GQA_GROUPS = NHEADS // NHEADS_KV

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SEQLENS        = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_KV_LENS        = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_BATCH          = 16
DEFAULT_DECODE_BATCHES = [1, 8, 16, 32]
DEFAULT_REPEATS        = 30
DEFAULT_WARMUP         = 10
DEVICE                 = "cuda"

DTYPES = {"fp16": torch.float16}
_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1}

# ── Plot styling ──────────────────────────────────────────────────────────────
_IMPL_COLORS = {"FA3": "#e74c3c", "SDPA": "#3498db", "cuDNN": "#2ecc71", "Unfused": "#95a5a6"}
_DTYPE_LS    = {"fp16": "-",  "bf16": "--", "fp8": ":"}
_DTYPE_MK    = {"fp16": "o",  "bf16": "s",  "fp8": "^"}

# ── Hardware peaks (H200 SXM5, dense without sparsity) ───────────────────────
HW_PEAK_TFLOPS = {"fp16": 989.0, "bf16": 989.0, "fp8": 1979.0}
HW_PEAK_BW_GBS = 4800.0


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs — general seqlen_q / seqlen_k
# ─────────────────────────────────────────────────────────────────────────────

def _flops_fwd(batch, seqlen_q, seqlen_k, causal):
    # Bottom-right causal: query i attends to i + seqlen_k - seqlen_q + 1 keys
    avg_sk = (2 * seqlen_k - seqlen_q + 1) / 2 if causal else seqlen_k
    return batch * NHEADS * 2 * seqlen_q * avg_sk * 2 * HEADDIM


def _flops_bwd(batch, seqlen_q, seqlen_k, causal):
    return 2.5 * _flops_fwd(batch, seqlen_q, seqlen_k, causal)


def _bytes_decode(batch, seqlen_k, dtype_bytes):
    """Bytes read/written for one decode step (seqlen_q=1).
    K and V cache dominate; Q and O are negligible at long context.
    """
    return dtype_bytes * batch * HEADDIM * (
        NHEADS * 1 +            # Q  read
        NHEADS_KV * seqlen_k +  # K  read
        NHEADS_KV * seqlen_k +  # V  read
        NHEADS * 1              # O  write
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tensor helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_qkv(batch, seqlen_q, seqlen_k, dtype, grad=False):
    """Returns (q, k, v) in FA3 layout (B, S, H, D)."""
    q = torch.randn(batch, seqlen_q, NHEADS,    HEADDIM, device=DEVICE, dtype=dtype)
    k = torch.randn(batch, seqlen_k, NHEADS_KV, HEADDIM, device=DEVICE, dtype=dtype)
    v = torch.randn(batch, seqlen_k, NHEADS_KV, HEADDIM, device=DEVICE, dtype=dtype)
    if grad:
        q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
    return q, k, v


def _prepare_sdpa(q, k, v):
    """Transpose FA3 layout (B,S,H,D) → SDPA layout (B,H,S,D). GQA handled by enable_gqa."""
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Attention wrappers  (all accept FA3-layout tensors)
# ─────────────────────────────────────────────────────────────────────────────

def _fa3(q, k, v, causal):
    return _fa3_func(q, k, v, causal=causal)


def _fa3_decode(q, k_cache, v_cache, seqlen_k):
    """Decode via flash_attn_with_kvcache with num_splits=0 (auto split-KV heuristic)."""
    return _fa3_kvcache_func(q, k_cache, v_cache,
                             cache_seqlens=seqlen_k,
                             causal=False,
                             num_splits=0,
                             pack_gqa=True)


def _sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)


def _unfused_decode(q, k, v, causal=False):
    """Unfused reference for decode (SDPA layout B,H,S,D, GQA K/V expanded here).
    Does three separate kernel launches: QK^T, softmax, AV.
    """
    k = k.repeat_interleave(GQA_GROUPS, dim=1)
    v = v.repeat_interleave(GQA_GROUPS, dim=1)
    scale = q.shape[-1] ** -0.5
    s = torch.matmul(q * scale, k.transpose(-1, -2))       # (B, H, 1, S)
    p = torch.softmax(s.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)                               # (B, H, 1, D)


def _try_cudnn_decode_fn(batch, seqlen_k, dtype):
    """Build a cuDNN attention graph for one (batch, seqlen_k, dtype) and return
    a zero-arg callable that executes it, or None if cuDNN is unavailable /
    doesn't support this config.

    Uses graph.sdpa_fp8() for fp8 (matches hopper/benchmark_flash_attention_fp8.py)
    and graph.sdpa() for fp16/bf16.
    Tensors are in cuDNN layout: (B, H, S, D).
    """
    try:
        import cudnn
    except ImportError:
        return None

    _cudnn_type = {
        torch.float16:       cudnn.data_type.HALF,
        torch.bfloat16:      cudnn.data_type.BFLOAT16,
        torch.float32:       cudnn.data_type.FLOAT,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
    }

    is_fp8 = (dtype == torch.float8_e4m3fn)

    try:
        # cuDNN layout: (B, H, S, D)
        q_gpu = torch.randn(batch, NHEADS,    1,        HEADDIM, device=DEVICE,
                            dtype=torch.float16).to(dtype)
        k_gpu = torch.randn(batch, NHEADS_KV, seqlen_k, HEADDIM, device=DEVICE,
                            dtype=torch.float16).to(dtype)
        v_gpu = torch.randn(batch, NHEADS_KV, seqlen_k, HEADDIM, device=DEVICE,
                            dtype=torch.float16).to(dtype)
        # fp8 SDPA outputs in fp16; fp16/bf16 outputs match input dtype
        o_gpu = torch.empty(batch, NHEADS, 1, HEADDIM, device=DEVICE,
                            dtype=torch.float16 if is_fp8 else dtype)

        graph = cudnn.pygraph(
            io_data_type=_cudnn_type[dtype],
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        q_ = graph.tensor_like(q_gpu)
        k_ = graph.tensor_like(k_gpu)
        v_ = graph.tensor_like(v_gpu)

        sdpa_kwargs = dict(
            name="sdpa_decode",
            q=q_, k=k_, v=v_,
            is_inference=True,
            attn_scale=HEADDIM ** -0.5,
            use_causal_mask=False,
        )
        if is_fp8:
            # Per-tensor descales (scalar tensors, value=1.0 = no rescaling)
            one = torch.ones(1, device=DEVICE, dtype=torch.float32)
            descale_q_ = graph.tensor_like(one)
            descale_k_ = graph.tensor_like(one)
            descale_v_ = graph.tensor_like(one)
            sdpa_kwargs.update(descale_q=descale_q_, descale_k=descale_k_, descale_v=descale_v_)

        o_, _ = graph.sdpa(**sdpa_kwargs)
        o_.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
        if is_fp8:
            o_.set_data_type(cudnn.data_type.HALF)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        pack = {q_: q_gpu, k_: k_gpu, v_: v_gpu, o_: o_gpu}
        if is_fp8:
            pack.update({descale_q_: one, descale_k_: one, descale_v_: one})

        workspace = torch.empty(graph.get_workspace_size(), device=DEVICE, dtype=torch.uint8)
        return lambda: graph.execute(pack, workspace)

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

def _bench_fwd(fn, batch, seqlen_q, seqlen_k, dtype, causal, warmup, repeats):
    q, k, v = _make_qkv(batch, seqlen_q, seqlen_k, dtype)
    if fn is _sdpa:
        q, k, v = _prepare_sdpa(q, k, v)   # done once, outside the timed loop
    ms = do_bench(lambda: fn(q, k, v, causal), warmup=warmup, rep=repeats)
    return ms


def _bench_bwd(fn, batch, seqlen_q, seqlen_k, dtype, causal, warmup, repeats):
    q, k, v = _make_qkv(batch, seqlen_q, seqlen_k, dtype, grad=True)
    if fn is _sdpa:
        q, k, v = _prepare_sdpa(q, k, v)
        # SDPA needs grad on the pre-prepared tensors
        q  = q.detach().requires_grad_(True)
        k  = k.detach().requires_grad_(True)
        v  = v.detach().requires_grad_(True)
    out  = fn(q, k, v, causal)
    dout = torch.randn_like(out)

    def _bwd():
        q.grad = k.grad = v.grad = None
        out.backward(dout, retain_graph=True)

    ms = do_bench(_bwd, warmup=warmup, rep=repeats)
    return ms


def _bench_fwd_fp8(batch, seqlen_q, seqlen_k, causal, warmup, repeats, pack_gqa=None):
    """FA3 fp8 forward — descale shape is (batch, nheads_kv)."""
    to_fp8 = lambda t: t.to(torch.float8_e4m3fn)
    q = to_fp8(torch.randn(batch, seqlen_q, NHEADS,    HEADDIM, device=DEVICE, dtype=torch.bfloat16))
    k = to_fp8(torch.randn(batch, seqlen_k, NHEADS_KV, HEADDIM, device=DEVICE, dtype=torch.bfloat16))
    v = to_fp8(torch.randn(batch, seqlen_k, NHEADS_KV, HEADDIM, device=DEVICE, dtype=torch.bfloat16))
    q_d = torch.ones(batch, NHEADS_KV, device=DEVICE, dtype=torch.float32)
    k_d = torch.ones(batch, NHEADS_KV, device=DEVICE, dtype=torch.float32)
    v_d = torch.ones(batch, NHEADS_KV, device=DEVICE, dtype=torch.float32)
    ms = do_bench(
        lambda: _fa3_func(q, k, v, causal=causal, pack_gqa=pack_gqa,
                          q_descale=q_d, k_descale=k_d, v_descale=v_d),
        warmup=warmup, rep=repeats,
    )
    return ms


# ─────────────────────────────────────────────────────────────────────────────
# Collection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_impl(fn, impl_name, batch, seqlen_q, seqlen_k, dtype_name, dtype,
              causal, warmup, repeats, include_bwd):
    """Run one (impl, dtype, causal, seqlen) configuration; return (row_fwd, row_bwd)."""
    ms_f = _bench_fwd(fn, batch, seqlen_q, seqlen_k, dtype, causal, warmup, repeats)
    tf_f = _flops_fwd(batch, seqlen_q, seqlen_k, causal) / (ms_f * 1e-3) * 1e-12

    ms_b = tf_b = None
    if include_bwd:
        ms_b = _bench_bwd(fn, batch, seqlen_q, seqlen_k, dtype, causal, warmup, repeats)
        tf_b = _flops_bwd(batch, seqlen_q, seqlen_k, causal) / (ms_b * 1e-3) * 1e-12

    bwd_str = f"  bwd {ms_b:6.2f}ms {tf_b:5.1f}TF" if ms_b else ""
    tag = f"seqlen={seqlen_k:6d}" if seqlen_q == seqlen_k else f"kv_len={seqlen_k:6d} sq=1"
    print(f"  {impl_name:4s} {dtype_name} causal={causal} {tag}"
          f"  fwd {ms_f:6.2f}ms {tf_f:5.1f}TF{bwd_str}")

    return (seqlen_k, ms_f, tf_f), ((seqlen_k, ms_b, tf_b) if ms_b else None)


# ─────────────────────────────────────────────────────────────────────────────
# Prefill collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_prefill(seqlens, batch, repeats, warmup, include_sdpa, include_bwd, include_fp8):
    """seqlen_q = seqlen_k; results keyed (impl, dtype, causal, pass)."""
    impls = [("FA3", _fa3)]
    if include_sdpa:
        impls.append(("SDPA", _sdpa))

    results = {}
    print("\n── Prefill ──────────────────────────────────────────────────────────────────")

    for causal in [False, True]:
        for dtype_name, dtype in DTYPES.items():
            for impl_name, fn in impls:
                rows_fwd, rows_bwd = [], []
                for seqlen in seqlens:
                    try:
                        rf, rb = _run_impl(fn, impl_name, batch, seqlen, seqlen,
                                           dtype_name, dtype, causal, warmup, repeats, include_bwd)
                        rows_fwd.append(rf)
                        if rb: rows_bwd.append(rb)
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        print(f"  SKIP {impl_name} {dtype_name} causal={causal} seqlen={seqlen}: {e}")
                        break
                results[(impl_name, dtype_name, causal, "fwd")] = rows_fwd
                if include_bwd:
                    results[(impl_name, dtype_name, causal, "bwd")] = rows_bwd

    if include_fp8:
        for causal in [False, True]:
            rows = []
            for seqlen in seqlens:
                try:
                    ms = _bench_fwd_fp8(batch, seqlen, seqlen, causal, warmup, repeats)
                    tf = _flops_fwd(batch, seqlen, seqlen, causal) / (ms * 1e-3) * 1e-12
                    rows.append((seqlen, ms, tf))
                    print(f"  FA3  fp8  causal={causal} seqlen={seqlen:6d}  fwd {ms:6.2f}ms {tf:5.1f}TF  [fwd only]")
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f"  SKIP FA3 fp8 causal={causal} seqlen={seqlen}: {e}")
                    break
            results[("FA3", "fp8", causal, "fwd")] = rows

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Decode collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_decode(kv_lens, batch_sizes, repeats, warmup, include_fp8, include_naive=True):
    """seqlen_q=1, causal=False; results keyed (impl, dtype, batch).

    Decode uses causal=False: the single query token always attends to all KV
    positions, so the causal mask is a no-op and causal=True would only select
    a suboptimal kernel path.

    Implementations benchmarked:
      FA3     — always (primary)
      cuDNN   — attempted; skipped per (dtype, kv_len) if unsupported
      Unfused — explicit QK^T + softmax + AV (reference for fusion benefit)
    SDPA excluded: timing is constant regardless of kv_len on PyTorch SDPA for
    seqlen_q=1 (launch-overhead dominated, not reading HBM).
    """
    results = {}
    print("\n── Decode (seqlen_q=1, causal=False) ────────────────────────────────────────")

    # Check cuDNN availability once
    try:
        import cudnn as _cudnn_mod
        _cudnn_present = True
    except ImportError:
        _cudnn_present = False
        print("  cuDNN python bindings not found — skipping cuDNN decode")

    def _record(rows, impl, dtype_name, batch, kv_len, ms):
        gbps = _bytes_decode(batch, kv_len, _DTYPE_BYTES[dtype_name]) / (ms * 1e-3) * 1e-9
        rows.append((kv_len, ms, gbps))
        print(f"  {impl:7s} {dtype_name} bs={batch:3d} kv_len={kv_len:6d}"
              f"  {ms:7.3f}ms  {gbps:6.0f}GB/s")

    for batch in batch_sizes:
        for dtype_name, dtype in DTYPES.items():

            # ── FA3 (flash_attn_with_kvcache, num_splits=0 auto split-KV) ────
            rows = []
            for kv_len in kv_lens:
                try:
                    q, k, v = _make_qkv(batch, 1, kv_len, dtype)
                    ms = do_bench(lambda q=q, k=k, v=v, kv_len=kv_len:
                                  _fa3_decode(q, k, v, kv_len),
                                  warmup=warmup, rep=repeats)
                    _record(rows, "FA3", dtype_name, batch, kv_len, ms)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f"  SKIP FA3 {dtype_name} bs={batch} kv_len={kv_len}: {e}"); break
            results[("FA3", dtype_name, batch)] = rows

            # ── cuDNN ─────────────────────────────────────────────────────────
            if _cudnn_present:
                rows = []
                for kv_len in kv_lens:
                    fn = _try_cudnn_decode_fn(batch, kv_len, dtype)
                    if fn is None:
                        print(f"  cuDNN   {dtype_name} bs={batch} kv_len={kv_len}: not supported"); break
                    ms = do_bench(fn, warmup=warmup, rep=repeats)
                    _record(rows, "cuDNN", dtype_name, batch, kv_len, ms)
                if rows:
                    results[("cuDNN", dtype_name, batch)] = rows

            # ── Unfused reference ─────────────────────────────────────────────
            if include_naive:
                rows = []
                for kv_len in kv_lens:
                    try:
                        q, k, v = _make_qkv(batch, 1, kv_len, dtype)
                        q_, k_, v_ = _prepare_sdpa(q, k, v)   # pre-expand outside timing
                        ms = do_bench(lambda: _unfused_decode(q_, k_, v_), warmup=warmup, rep=repeats)
                        _record(rows, "Unfused", dtype_name, batch, kv_len, ms)
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                        print(f"  SKIP Unfused {dtype_name} bs={batch} kv_len={kv_len}: {e}"); break
                results[("Unfused", dtype_name, batch)] = rows

        # ── fp8 (FA3 + cuDNN) ─────────────────────────────────────────────────
        if include_fp8:
            rows = []
            for kv_len in kv_lens:
                try:
                    ms = _bench_fwd_fp8(batch, 1, kv_len, False, warmup, repeats, pack_gqa=True)
                    _record(rows, "FA3", "fp8", batch, kv_len, ms)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f"  SKIP FA3 fp8 bs={batch} kv_len={kv_len}: {e}"); break
            results[("FA3", "fp8", batch)] = rows

            # cuDNN fp8 SDPA requires Blackwell (SM100); skip on SM90/H200

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def _active_combos(results):
    """Return (impl, dtype) pairs that have at least one data point."""
    return list(dict.fromkeys(
        (impl, dt) for (impl, dt, *_), rows in results.items() if rows
    ))


def print_prefill_tables(results, seqlens):
    combos    = _active_combos(results)
    all_pass  = list(dict.fromkeys(k[3] for k in results))
    col_names = [f"{impl}/{dt}" for impl, dt in combos]
    col_w     = max(12, max(len(c) for c in col_names) + 2)

    for causal in [False, True]:
        for pass_ in all_pass:
            note = "  [fp8: fwd/FA3 only]" if pass_ == "fwd" else ""
            print(f"\n{'═'*80}\n  {pass_.upper()}  causal={causal}  (TFLOPS){note}\n{'═'*80}")
            hdr = f"{'seqlen':>8}" + "".join(f"{c:>{col_w}}" for c in col_names)
            print(hdr); print("─" * len(hdr))
            for s in seqlens:
                row = f"{s:>8}"
                for impl, dt in combos:
                    data  = results.get((impl, dt, causal, pass_), [])
                    match = next((tf for sl, ms, tf in data if sl == s), None)
                    row  += f"{match:>{col_w}.1f}" if match is not None else f"{'—':>{col_w}}"
                print(row)


def print_decode_tables(results, kv_lens, batch_sizes):
    combos    = _active_combos(results)
    col_names = [f"{impl}/{dt}" for impl, dt in combos]
    col_w     = max(12, max(len(c) for c in col_names) + 2)

    metrics = [
        ("Latency (ms)",   lambda r: r[1], ".3f"),
        ("Bandwidth GB/s", lambda r: r[2], ".0f"),
    ]

    for batch in batch_sizes:
        for label, getter, fmt in metrics:
            print(f"\n{'═'*80}\n  DECODE  batch={batch}  seqlen_q=1  — {label}\n{'═'*80}")
            hdr = f"{'kv_len':>8}" + "".join(f"{c:>{col_w}}" for c in col_names)
            print(hdr); print("─" * len(hdr))
            for kv in kv_lens:
                row = f"{kv:>8}"
                for impl, dt in combos:
                    data  = results.get((impl, dt, batch), [])
                    match = next((getter(r) for r in data if r[0] == kv), None)
                    row  += f"{match:>{col_w}{fmt}}" if match is not None else f"{'—':>{col_w}}"
                print(row)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _axis_setup(ax, xs, xlabel, ylabel, log_x=True):
    import matplotlib.ticker as ticker
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xticks(xs)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, which="both", alpha=0.25)


def plot_prefill(results, seqlens, include_bwd, save_path, gpu_name):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot"); return

    combos  = _active_combos(results)
    passes  = ["fwd"] + (["bwd"] if include_bwd else [])
    nrows   = len(passes)

    for metric, ylabel, suffix in [("tflops", "TFLOPS", ""), ("ms", "Latency (ms)", "_latency")]:
        fig, axes = plt.subplots(nrows, 2, figsize=(13, 5 * nrows), squeeze=False)
        fig.suptitle(f"FA3 Prefill — {MODEL}\n"
                     f"nheads={NHEADS}, nheads_kv={NHEADS_KV}, headdim={HEADDIM}, batch={DEFAULT_BATCH}\n{gpu_name}",
                     fontsize=11, fontweight="bold")

        for row, pass_ in enumerate(passes):
            for col, causal in enumerate([False, True]):
                ax = axes[row][col]
                ax.set_title(f"{'Forward' if pass_=='fwd' else 'Backward'} — "
                             f"{'Causal' if causal else 'Non-causal'}", fontsize=11)
                _axis_setup(ax, seqlens, "Sequence length", ylabel)
                for impl, dt in combos:
                    data = results.get((impl, dt, causal, pass_), [])
                    if not data: continue
                    xs = [s  for s, ms, tf in data]
                    ys = [tf for s, ms, tf in data] if metric == "tflops" else [ms for s, ms, tf in data]
                    ax.plot(xs, ys, label=f"{impl} {dt}",
                            color=_IMPL_COLORS.get(impl, "gray"),
                            linestyle=_DTYPE_LS.get(dt, "-"),
                            marker=_DTYPE_MK.get(dt, "o"),
                            linewidth=2, markersize=6)
                if metric == "tflops":
                    active_dtypes = {dt for _, dt in combos if any(results.get((impl, dt, causal, pass_), []) for impl, _ in combos)}
                    drawn = set()
                    for dt in active_dtypes:
                        peak = HW_PEAK_TFLOPS.get(dt)
                        if peak and peak not in drawn:
                            ax.axhline(peak, color="black", linestyle="--", linewidth=1, alpha=0.5,
                                       label=f"Peak {dt} ({peak:.0f} TFLOPS)")
                            drawn.add(peak)
                ax.legend(fontsize=9, loc="upper left")

        fig.tight_layout()
        out = save_path.replace(".png", f"{suffix}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  prefill {'TFLOPS' if not suffix else 'latency'} → {out}")


def plot_decode(results, kv_lens, batch_sizes, save_path, gpu_name):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    combos = _active_combos(results)
    nbatch = len(batch_sizes)

    # (tuple_idx, ylabel, file_suffix)  — tuple is (kv_len, ms, gbps)
    decode_metrics = [
        (1, "Latency (ms)",     "_decode_latency"),
        (2, "Memory BW (GB/s)", "_decode_bw"),
    ]

    for idx, ylabel, suffix in decode_metrics:
        fig, axes = plt.subplots(1, nbatch, figsize=(6 * nbatch, 5), squeeze=False)
        fig.suptitle(f"FA3 Decode (seqlen_q=1) — {MODEL}\n"
                     f"nheads={NHEADS}, nheads_kv={NHEADS_KV}, headdim={HEADDIM}\n{gpu_name}",
                     fontsize=11, fontweight="bold")

        for col, batch in enumerate(batch_sizes):
            ax = axes[0][col]
            ax.set_title(f"batch = {batch}", fontsize=11)
            _axis_setup(ax, kv_lens, "KV-cache length", ylabel)
            for impl, dt in combos:
                data = results.get((impl, dt, batch), [])
                if not data: continue
                xs = [r[0]   for r in data]
                ys = [r[idx] for r in data]
                ax.plot(xs, ys, label=f"{impl} {dt}",
                        color=_IMPL_COLORS.get(impl, "gray"),
                        linestyle=_DTYPE_LS.get(dt, "-"),
                        marker=_DTYPE_MK.get(dt, "o"),
                        linewidth=2, markersize=6)
            if ylabel == "Memory BW (GB/s)":
                ax.axhline(HW_PEAK_BW_GBS, color="black", linestyle="--",
                           linewidth=1, alpha=0.5,
                           label=f"Peak BW ({HW_PEAK_BW_GBS:.0f} GB/s)")
            ax.legend(fontsize=9, loc="upper left")

        fig.tight_layout()
        out = save_path.replace(".png", f"{suffix}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  decode {ylabel} → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Profiler
# ─────────────────────────────────────────────────────────────────────────────

def profile_config(label, fn_zero_arg, batch, seqlen_q, seqlen_k, causal,
                   trace_path=None, warmup=5, active=3):
    import torch.profiler as P

    for _ in range(warmup):
        fn_zero_arg()
    torch.cuda.synchronize()

    with P.profile(
        activities=[P.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(active):
            with P.record_function(label):
                fn_zero_arg()

    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"  {label}  →  {trace_path}")
    else:
        print(f"  {label}  (no --profile-trace set, trace not saved)")


def run_profiles(args, run_fp8):
    """Profile representative prefill and decode configurations."""
    seqlen   = args.profile_seqlen or args.seqlens[-1]
    kv_len   = args.profile_kv_len or args.kv_lens[-1]
    batch    = args.batch
    trace_pfx = args.profile_trace   # None → no file export

    def _trace(tag):
        if trace_pfx is None:
            return None
        safe = tag.replace(" ", "_").replace("=", "").replace("/", "_")
        return f"{trace_pfx}_{safe}.json"

    print(f"\n{'━'*80}")
    print(f"  PROFILING  seqlen={seqlen}  kv_len={kv_len}  batch={batch}")
    print(f"{'━'*80}")

    # ── Prefill ───────────────────────────────────────────────────────────────
    for dtype_name, dtype in DTYPES.items():
        for causal in [False, True]:
            q, k, v = _make_qkv(batch, seqlen, seqlen, dtype)
            tag = f"FA3 {dtype_name} prefill seqlen={seqlen} causal={causal}"
            profile_config(tag, lambda q=q, k=k, v=v: _fa3(q, k, v, causal),
                           batch, seqlen, seqlen, causal, _trace(tag))

    if run_fp8:
        for causal in [False, True]:
            to_fp8 = lambda t: t.to(torch.float8_e4m3fn)
            q = to_fp8(torch.randn(batch, seqlen, NHEADS,    HEADDIM, device=DEVICE, dtype=torch.bfloat16))
            k = to_fp8(torch.randn(batch, seqlen, NHEADS_KV, HEADDIM, device=DEVICE, dtype=torch.bfloat16))
            v = to_fp8(torch.randn(batch, seqlen, NHEADS_KV, HEADDIM, device=DEVICE, dtype=torch.bfloat16))
            qd = torch.ones(batch, NHEADS_KV, device=DEVICE, dtype=torch.float32)
            kd = torch.ones(batch, NHEADS_KV, device=DEVICE, dtype=torch.float32)
            vd = torch.ones(batch, NHEADS_KV, device=DEVICE, dtype=torch.float32)
            tag = f"FA3 fp8 prefill seqlen={seqlen} causal={causal}"
            profile_config(tag,
                           lambda q=q, k=k, v=v, qd=qd, kd=kd, vd=vd:
                               _fa3_func(q, k, v, causal=causal,
                                         q_descale=qd, k_descale=kd, v_descale=vd),
                           batch, seqlen, seqlen, causal, _trace(tag))

    # ── Decode (flash_attn_with_kvcache, num_splits=0) ────────────────────────
    dec_batch = args.decode_batches[0]
    for dtype_name, dtype in DTYPES.items():
        q, k, v = _make_qkv(dec_batch, 1, kv_len, dtype)
        tag = f"FA3 {dtype_name} decode kv_len={kv_len} bs={dec_batch}"
        profile_config(tag,
                       lambda q=q, k=k, v=v, kv_len=kv_len: _fa3_decode(q, k, v, kv_len),
                       dec_batch, 1, kv_len, False, _trace(tag))

    if run_fp8:
        to_fp8 = lambda t: t.to(torch.float8_e4m3fn)
        q = to_fp8(torch.randn(dec_batch, 1,      NHEADS,    HEADDIM, device=DEVICE, dtype=torch.bfloat16))
        k = to_fp8(torch.randn(dec_batch, kv_len, NHEADS_KV, HEADDIM, device=DEVICE, dtype=torch.bfloat16))
        v = to_fp8(torch.randn(dec_batch, kv_len, NHEADS_KV, HEADDIM, device=DEVICE, dtype=torch.bfloat16))
        qd = torch.ones(dec_batch, NHEADS_KV, device=DEVICE, dtype=torch.float32)
        kd = torch.ones(dec_batch, NHEADS_KV, device=DEVICE, dtype=torch.float32)
        vd = torch.ones(dec_batch, NHEADS_KV, device=DEVICE, dtype=torch.float32)
        tag = f"FA3 fp8 decode kv_len={kv_len} bs={dec_batch}"
        profile_config(tag,
                       lambda q=q, k=k, v=v, qd=qd, kd=kd, vd=vd, kv_len=kv_len:
                           _fa3_kvcache_func(q, k, v, causal=False,
                                             cache_seqlens=kv_len, num_splits=0,
                                             pack_gqa=True,
                                             q_descale=qd, k_descale=kd, v_descale=vd),
                       dec_batch, 1, kv_len, False, _trace(tag))


def main():
    global DEFAULT_BATCH, MODEL, NHEADS, NHEADS_KV, HEADDIM, GQA_GROUPS
    parser = argparse.ArgumentParser(
        description="Benchmark FA3 on Llama 3 attention dimensions."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(_MODELS),
                        help="Model config: " + ", ".join(
                            f"{k} (nheads={v['nheads']}, nheads_kv={v['nheads_kv']}, headdim={v['headdim']})"
                            for k, v in _MODELS.items()))
    parser.add_argument("--seqlens",        nargs="+", type=int, default=DEFAULT_SEQLENS)
    parser.add_argument("--kv-lens",        nargs="+", type=int, default=DEFAULT_KV_LENS,
                        help="KV-cache lengths for decode mode")
    parser.add_argument("--batch",          type=int,  default=DEFAULT_BATCH,
                        help="Batch size for prefill")
    parser.add_argument("--decode-batches", nargs="+", type=int, default=DEFAULT_DECODE_BATCHES,
                        help="Batch sizes for decode mode")
    parser.add_argument("--repeats",        type=int,  default=DEFAULT_REPEATS)
    parser.add_argument("--warmup",         type=int,  default=DEFAULT_WARMUP)
    parser.add_argument("--no-bwd",         action="store_true")
    parser.add_argument("--no-sdpa",        action="store_true")
    parser.add_argument("--no-fp8",         action="store_true")
    parser.add_argument("--no-naive",       action="store_true",
                        help="Skip Unfused (naive QK^T+softmax+AV) decode baseline")
    parser.add_argument("--no-decode",        action="store_true")
    parser.add_argument("--no-prefill",       action="store_true")
    parser.add_argument("--save",             default="fa3_benchmark.png")
    # profiler
    parser.add_argument("--profile",          action="store_true",
                        help="Run torch.profiler after benchmarks and print per-kernel table")
    parser.add_argument("--profile-seqlen",   type=int, default=None,
                        help="Seqlen for prefill profiling (default: last --seqlens value)")
    parser.add_argument("--profile-kv-len",   type=int, default=None,
                        help="KV-cache length for decode profiling (default: last --kv-lens value)")
    parser.add_argument("--profile-trace",    type=str, default=None,
                        metavar="PREFIX",
                        help="Export Chrome traces as PREFIX_<config>.json (e.g. --profile-trace traces/fa3)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available"); return

    _cfg = _MODELS[args.model]
    MODEL      = _cfg["label"]
    NHEADS     = _cfg["nheads"]
    NHEADS_KV  = _cfg["nheads_kv"]
    HEADDIM    = _cfg["headdim"]
    GQA_GROUPS = NHEADS // NHEADS_KV

    DEFAULT_BATCH = args.batch
    gpu_name      = torch.cuda.get_device_name()

    fp8_ok  = torch.cuda.get_device_capability()[0] >= 9
    run_fp8 = not args.no_fp8 and fp8_ok
    if not args.no_fp8 and not fp8_ok:
        print("NOTE: fp8 requires SM90+ — skipping.")

    print(f"GPU    : {gpu_name}")
    print(f"Model  : {MODEL}  nheads={NHEADS}, nheads_kv={NHEADS_KV}, headdim={HEADDIM}")
    print(f"Dtypes : fp16, bf16" + (", fp8 (fwd/FA3 only)" if run_fp8 else ""))

    prefill_results = decode_results = None

    if not args.no_prefill:
        print(f"\nPrefill: batch={args.batch}, seqlens={args.seqlens}")
        prefill_results = collect_prefill(
            seqlens=args.seqlens, batch=args.batch,
            repeats=args.repeats, warmup=args.warmup,
            include_sdpa=not args.no_sdpa,
            include_bwd=not args.no_bwd,
            include_fp8=run_fp8,
        )
        print_prefill_tables(prefill_results, args.seqlens)

    if not args.no_decode:
        print(f"\nDecode: seqlen_q=1, causal=False, kv_lens={args.kv_lens}, batches={args.decode_batches}")
        decode_results = collect_decode(
            kv_lens=args.kv_lens, batch_sizes=args.decode_batches,
            repeats=args.repeats, warmup=args.warmup,
            include_fp8=run_fp8,
            include_naive=not args.no_naive,
        )
        print_decode_tables(decode_results, args.kv_lens, args.decode_batches)

    print("\nSaving plots...")
    if prefill_results:
        plot_prefill(prefill_results, args.seqlens, not args.no_bwd, args.save, gpu_name)
    if decode_results:
        plot_decode(decode_results, args.kv_lens, args.decode_batches, args.save, gpu_name)

    if args.profile:
        run_profiles(args, run_fp8)


if __name__ == "__main__":
    main()
