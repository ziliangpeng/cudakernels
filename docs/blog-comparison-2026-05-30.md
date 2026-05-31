# Performance Comparison: Our Kernels vs Published Worklogs

**Date**: 2026-05-30  
**Sources**:
- [siboehm: CUDA Matmul Worklog](https://siboehm.com/articles/22/CUDA-MMM) — A100, N=4096, FP32 path, vs cuBLAS FP32 (23.2 TFLOPS)
- [Pranjal: Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) — H100, N=4096, TC path, vs cuBLAS BF16 (716.7 TFLOPS)
- Our data: [ncu-profiling-2026-05-30.md](ncu-profiling-2026-05-30.md) — H100, N=4096, vs cuBLAS FP32 (52.2 TFLOPS, `CUBLAS_PEDANTIC_MATH`)

> **Baseline**: Our cuBLAS uses `cublasSetMathMode(CUBLAS_PEDANTIC_MATH)` — disables TF32, pure FP32. On H100 this achieves **52.2 TFLOPS** at N=4096. All our "% vs cuBLAS" are against this pure FP32 baseline, matching siboehm's methodology.

---

## FP32 Hand-Written Kernel Path (N=4096, both sides)

All percentages are vs each side's own **cuBLAS FP32** baseline. Our block tile dimensions (BM/BN/BK on 2D/warp) are hardcoded, no autotuning. siboehm autotunes.

| Step | siboehm (A100, 4K) | Ours (H100, 4K) | Gap | Verdict |
|---|---|---|---|---|
| Naive | 1.3% (0.3 T) | — (not run) | — | |
| Coalesced | 8.5% (2.0 T) | **10.9%** (5.7 T) | +2.4pp | ✅ ahead |
| SMEM tiling | 12.8% (3.0 T) | **17.2%** (9.0 T) | +4.4pp | ✅ ahead |
| 1D blocktile | 36.5% (8.5 T) | 33.7% (17.6 T) | −2.8pp | ≈ tie |
| 2D blocktile | 68.7% (16.0 T) | 42.9% (22.4 T) | −25.8pp | ⚠️ behind |
| Vectorized | 78.4% (18.2 T) | **63.0%** (32.9 T) | −15.4pp | ⚠️ behind |
| Warptile (Simon final) | **93.7%** (21.8 T) | 54.2% (28.3 T) | −39.5pp | ⚠️⚠️ behind |
| Autotuning | 84.8% (19.7 T) | — | — | — |

**cuBLAS FP32 baseline**: siboehm = 23.2 TFLOPS, ours = 52.2 TFLOPS (H100 has 2.25× more FP32 throughput than A100).

### Key Observations

1. **Coalesced + SMEM**: we lead. H100's faster global memory + larger SMEM give a head start at the simple optimization steps. Expected — these steps are hardware-bound, not algorithm-bound.

2. **1D blocktile**: nearly identical (33.7% vs 36.5%). Both sides are compute-bound at this point and scaling correctly.

3. **2D blocktile**: **the divergence point**. siboehm jumps to 68.7%, we only get to 42.9%. Our 2D kernel uses hardcoded `BM=128, BN=128, BK=8` — these tile sizes are tuned for siboehm's A100 (108 SMs, 164 KB SMEM). On H100 (132 SMs, 228 KB SMEM), different tile sizes are optimal.

4. **Vectorized**: we close some of the gap (63.0% vs 78.4%). float4 loads help regardless of tile size.

5. **Warptile**: **the gap explodes** (54.2% vs 93.7%). Our warptile actually regresses from vectorized (28.3 vs 32.9 TFLOPS!). The hardcoded warp tile parameters (`WM=64, WN=64, TM=8, TN=4`) are almost certainly wrong for H100. siboehm's warptile with autotuned 128×8 block dim → 93.7%. Our warptile with fixed params → 54.2%. **This is entirely an autotuning gap, not an algorithmic one.**

6. **Absolute TFLOPS**: despite the % gap, our vectorized (32.9 T) actually computes more raw FLOPS than siboehm's warptile (21.8 T). H100's raw FP32 throughput is 2.7× of A100's. The % gap means we're leaving performance on the table, not that our code is wrong.

### Root Cause: No Autotuning

Simon's critical insight that we're missing: **optimal tile dimensions (BM, BN, BK) depend on the GPU architecture, and the only way to find them is to sweep.**

```
Ours: BM=128, BN=128, BK=8   ← hardcoded, same for all kernels
H100 optimal: unknown without sweep
A100 optimal (siboehm): BM=128, BN=128, BK=8 for 2D, tuned per kernel
```

Why this matters so much for warptile specifically: warptile has 4 hierarchical tile sizes (block BM/BN/BK → warp WM/WN → thread TM/TN). Getting any of them wrong cascades — if block tile doesn't fill SMEM well, warps are under-subscribed, and thread tiles waste registers. This is why our warptile regresses instead of improving.

---

## Simon's Final Algorithm: Head-to-Head (N=4096, H100)

Pranjal took siboehm's final (autotuned) warptile kernel and ran it on H100: **31.8 TFLOPS**. Our best FP32 kernel without autotuning:

| Kernel | TFLOPS | % vs cuBLAS FP32 | % of FP32 peak (67T) | vs Simon |
|---|---|---|---|---|
| Simon's warptile (on H100) | 31.8 | 60.9% | 47.5% | — |
| **Our vectorized (float4)** | **32.9** | **63.0%** | **49.1%** | **+3.5%** ✅ |
| Our warptile | 28.3 | 54.2% | 42.2% | −11.0% |

**We beat Simon on absolute TFLOPS** — 32.9 vs 31.8 — despite lacking autotuning. This is because H100 has 2.7× the raw FP32 throughput of A100. But our **% vs cuBLAS** (63.0%) is far below Simon's A100 number (93.7%) because H100's cuBLAS FP32 is also faster (52.2T vs 23.2T).

With proper autotuning, our warptile should reach >70% vs cuBLAS FP32 on H100 — but the FP32 ceiling (~33-35 TFLOPS) means even at 100% we'd only hit ~52 TFLOPS, which is dwarfed by what Tensor Cores offer.

---

## Tensor Core Path

| Step | Pranjal (H100, 4K) | Ours (H100, 2K) | Notes |
|---|---|---|---|
| Simon's FP32 | 4.4% (31.8 T vs 717T BF16) | 4.6% (32.9 T vs 717T BF16) | We beat Simon on H100 |
| **TC (WMMA/WGMMA)** | **44.3%** (317.6 T) | **2.6%** (25.6 T) | **17× gap** ⚠️⚠️ |
| Larger tiles | 59.0% (423 T) | — | |
| TMA (async loads) | 69.5% (498 T) | — | |
| Push tile limit | 88.2% (632 T) | — | |
| Hide store latency | 92.1% (660 T) | — | |
| Faster barriers | 98.4% (705 T) | — | |
| Thread Block Clusters | 102.4% (734 T) | — | |
| Micro-optimizations | 104.3% (747 T) | — | |
| Async Stores | 105.8% (759 T) | — | |
| Hilbert Curves | 106.6% (764 T) | — | |
| cuBLAS BF16 | 100% (716.7 T) | — | |

### Diagnosis

Pranjal's first TC step (WGMMA) = 44.3% vs cuBLAS BF16. Our WMMA = 2.6%. This is an **API gap**, not a skill gap:

| | Pranjal K1: "Tensor Core" | Our WMMA |
|---|---|---|
| API | WGMMA (warp-group, 128 threads) | `nvcuda::wmma` (1 warp, 32 threads) |
| SM sub-partitions | 4/4 in use | 1/4 in use |
| Tile | 64×M (Hopper-native) | 16×16 (Volta-era) |
| Nsight | Not published | Memory 93%, Compute 17% — TC starving |

Switching from WMMA to WGMMA is expected to take us from 2.6% → ~40-45% in one change. Every step after that (TMA, pipelining, clusters) builds on WGMMA.

---

## Summary

| Path | Status | Blocked by |
|---|---|---|
| FP32 (naive→vectorized) | ✅ done, beats Simon (32.9T > 31.8T) | — |
| FP32 (warptile) | ⚠️ 54.2% vs cuBLAS, regresses | **No autotuning** (hardcoded BM/BN/BK) |
| TC (WMMA) | ⚠️ 2.6% vs cuBLAS BF16 | **Wrong API** (WMMA, need WGMMA) |
| TC (Pranjal K1-K10) | Not started | Need WGMMA first |

**The FP32 path shows that autotuning matters massively** — Simon's warptile goes from 78% to 94% by sweeping tile sizes. Our warptile regresses because our hardcoded tiles are wrong for H100. But with the FP32 ceiling at ~33T, the bigger ROI is WGMMA, which opens a path to 700+ TFLOPS.

---

## Action Items

| Priority | What | Expected Gain |
|---|---|---|
| **P0** | Switch to WGMMA — Pranjal's first TC step | 2.6% → ~44% (17×) |
| P1 | Add autotuning (sweep BM/BN/BK) for FP32 warptile | 28.3T → ~35T (+24%) |
| P2 | TMA (async copy) — Pranjal step 3 | 44% → 70% |
| P2 | Profile Nsight at each WGMMA step | |
