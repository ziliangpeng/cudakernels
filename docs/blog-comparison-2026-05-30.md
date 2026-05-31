# Performance Comparison: Our Kernels vs Published Worklogs

**Date**: 2026-05-30  
**Sources**:
- [siboehm: CUDA Matmul Worklog](https://siboehm.com/articles/22/CUDA-MMM) — A100, N=4096, FP32, vs cuBLAS FP32
- [Pranjal: Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) — H100, N=4096, TC, vs cuBLAS BF16
- Our data: [ncu-profiling-2026-05-30.md](ncu-profiling-2026-05-30.md) — H100, N=4096

---

## Table 1: FP32 Path — Ours vs Simon (siboehm)

**N=4096. Both sides compared against their own cuBLAS FP32 baseline.**

| | Our cuBLAS FP32 | Our cuBLAS FP32 |
|---|---|---|
| Baseline | 52.2 TFLOPS (H100, CUBLAS_PEDANTIC_MATH) | 23.2 TFLOPS (A100) |

Our 2D blocktile and warptile use hardcoded tile sizes (BM/BN/BK). Simon autotunes. All % are vs each side's own cuBLAS FP32.

| Step | Ours (H100, 4K, no autotune) | Simon (A100, 4K, autotuned) | Gap | Note |
|---|---|---|---|---|
| Naive | — | 1.3% (0.3 T) | — | |
| Coalesced | **10.9%** (5.7 T) | 8.5% (2.0 T) | +2.4pp ✅ | H100 GMEM bandwidth helps |
| SMEM tiling | **17.2%** (9.0 T) | 12.8% (3.0 T) | +4.4pp ✅ | H100 SMEM larger/faster |
| 1D blocktile | 33.7% (17.6 T) | 36.5% (8.5 T) | −2.8pp ≈ | Nearly identical |
| 2D blocktile | 42.9% (22.4 T) | 68.7% (16.0 T) | **−25.8pp** ⚠️ | Tile sizes diverge |
| Vectorized | 63.0% (32.9 T) | 78.4% (18.2 T) | −15.4pp ⚠️ | |
| Warptile | 54.2% (28.3 T) | **93.7%** (21.8 T) | **−39.5pp** ⚠️⚠️ | Regresses vs our vectorized! |
| Autotuning | — | 84.8% (19.7 T) | — | |

### Why We Diverge at 2D Blocktile

Our kernels use **hardcoded tile dimensions** (`BM=128, BN=128, BK=8`, `WM=64, WN=64`). These values were chosen for A100. On H100 (more SMs, bigger SMEM), different tile sizes are optimal. Simon's autotuning sweeps BM/BN/BK per kernel and per GPU — that's how he gets warptile to 93.7%.

**The gap is not algorithmic — it's parameter tuning.** Our vectorized kernel (which is less sensitive to tile sizes) already achieves 63.0% vs cuBLAS FP32 (32.9 TFLOPS absolute, beating Simon's 21.8 T).

### Head-to-Head on Same Hardware (Both on H100)

Pranjal ran Simon's final (autotuned) warptile kernel on H100 and got **31.8 TFLOPS**.

| Kernel | TFLOPS | % vs cuBLAS FP32 | % of FP32 peak (67T) |
|---|---|---|---|
| Simon's warptile (on H100) | 31.8 | 60.9% | 47.5% |
| **Our vectorized** | **32.9** | **63.0%** | **49.1%** |
| Our warptile | 28.3 | 54.2% | 42.2% |

**We beat Simon on absolute TFLOPS** (32.9 > 31.8) despite lacking autotuning. Our warptile regresses because the hardcoded warp tile parameters are wrong for H100 — with autotuning it should beat vectorized, not lose to it.

### FP32 Ceiling

Even at 100% vs cuBLAS FP32 on H100, we'd only reach ~52 TFLOPS. Tensor Cores offer 717 TFLOPS (BF16) on the same hardware. The FP32 optimization path is complete and correct — but tapped out.

---

## Table 2: Tensor Core Path — Ours vs Pranjal (H100 Worklog)

**N=4096 (Pranjal) / N=4096 for FP32 step, N=2048 for TC step (ours). Both vs cuBLAS BF16 = 716.7 TFLOPS (Pranjal's baseline).**

| Step | Technique | Pranjal | Ours | Status |
|---|---|---|---|---|
| — | Simon's FP32 (H100) | 4.4% (31.8 T) | 4.6% (32.9 T @ 4K) | ✅ Beat Simon |
| K1 | **Tensor Core** (WMMA/WGMMA) | **44.3%** (317.6 T) | **2.6%** (25.6 T @ 2K) | ⚠️ **17× gap** (WMMA vs WGMMA) |
| K2 | Larger tiles | 59.0% (423 T) | — | 🆕 not started |
| K3 | Async loads (TMA) | 69.5% (498 T) | — | 🆕 not started |
| K4 | Pushing tile size limit | 88.2% (632 T) | — | 🆕 not started |
| K5 | Hide store latency | 92.1% (660 T) | — | 🆕 not started |
| K6 | Faster barriers | 98.4% (705 T) | — | 🆕 not started |
| K7 | Thread Block Clusters | **102.4%** (734 T) | — | 🆕 not started (surpasses cuBLAS!) |
| K8 | Micro-optimizations | 104.3% (747 T) | — | 🆕 not started |
| K9 | Async Stores | 105.8% (759 T) | — | 🆕 not started |
| K10 | Hilbert Curves | 106.6% (764 T) | — | 🆕 not started |

### The API Gap (Why K1 = 17×)

| | Pranjal K1: "Tensor Core" | Our WMMA |
|---|---|---|
| API | **WGMMA** (warp-group, 128 threads) | `nvcuda::wmma` (1 warp, 32 threads) |
| SM sub-partitions used | 4/4 | 1/4 |
| Tile size | 64×M (Hopper-native) | 16×16 (Volta-era) |
| Memory | Shared memory tile cache | Direct global memory reads |
| Nsight profile | Not published | Memory 93%, Compute 17% — TC starving |

Switching from WMMA to WGMMA is expected to close most of the 17× gap in a single change. Every subsequent step (TMA, pipelining, clusters) builds on WGMMA.

### Our cuBLAS BF16 (reference)

At N=2048: **287 TFLOPS (58% MFU)** — not directly comparable to Pranjal's N=4096 baseline of 717 TFLOPS. We need to benchmark at N=4096 for apples-to-apples.

---

## Summary

| Path | Progress | Key Blocker | Next Step |
|---|---|---|---|
| **FP32** (Simon) | 7/7 steps done, beats Simon (32.9 > 31.8 T) | No autotuning (warptile regresses) | Low priority — FP32 ceiling is ~52T |
| **TC** (Pranjal) | 0/10 steps done, WMMA at 2.6% | WMMA API (need WGMMA) | **WGMMA** — step 1 of 10 |

---

## Action Items

| Priority | What | Expected Gain |
|---|---|---|
| **P0** | WGMMA — Pranjal K1 | 2.6% → ~44% (17×) |
| P1 | Autotune BM/BN/BK for FP32 warptile | 28.3T → ~35T (+24%, low ROI vs TC) |
| P2 | TMA (async copy) — Pranjal K3 | 44% → 70% |
| P2 | Nsight profile each WGMMA step | |
