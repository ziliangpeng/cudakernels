# Performance Comparison: Our Kernels vs Published Worklogs

**Date**: 2026-05-30 (updated 2026-06-01 with A6000 correction and A100 autotune)
**Source code**: [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) — verified from source

## Hardware and Methodology: How Simon and We Differ

| | Simon (siboehm) | Us |
|---|---|---|
| GPU | **NVIDIA A6000** (GA102) | A100-SXM4-40GB (GA100) |
| SMs | 84 | 108 |
| SM ratio | — | +29% |
| cuBLAS call | `cublasGemmEx(..., CUBLAS_GEMM_DEFAULT_TENSOR_OP)` | `cublasSgemm` + `CUBLAS_PEDANTIC_MATH` |
| cuBLAS math | **TF32 via Tensor Cores** | **Pure FP32** |
| cuBLAS TFLOPS @ 4096 | **23.2 T** | **18.6 T** |
| FLOP formula | `2 × M × N × K` | `2 × N³ − N²` |
| Benchmark | 10 repeats, mean elapsed | 100 iterations, median |
| N sizes | {128, 256, 512, 1K, 2K, 4K} | N=4096 |

**Why the cuBLAS baseline differs**: Simon uses `CUBLAS_GEMM_DEFAULT_TENSOR_OP` which lets cuBLAS use TF32 Tensor Cores on Ampere. We set `CUBLAS_PEDANTIC_MATH` to force pure FP32. Each % is vs its own cuBLAS.

---

## Table 1: FP32 Path — Ours vs Simon (N=4096)

| Step | Src | H100 (4K) | A100 (4K) | Simon (A6000, 4K) | TFLOPS Gap |
|---|---|---|---|---|---|
| Naive | [link](matmul_naive.cu) | 5.3 T (10.2%) | 2.4 T (12.8%) | 0.3 T (1.3%) | +2.1 T ✅ |
| Coalesced | [link](matmul_coalesced.cu) | 5.7 T (10.9%) | 3.0 T (16.0%) | 2.0 T (8.5%) | +1.0 T ≈ |
| SMEM tiling | [link](matmul_smem.cu) | 9.0 T (17.2%) | 5.3 T (28.4%) | 3.0 T (12.8%) | +2.3 T ✅ |
| 1D blocktile | [link](matmul_1d_blocktile.cu) | 17.6 T (33.7%) | 10.0 T (53.5%) | 8.5 T (36.5%) | +1.5 T |
| **1D blocktile (autotuned)** | [link](matmul_1d_blocktile.cu) | **19.3 T (36.9%)** | **11.2 T (59.9%)** | — | — |
| 2D blocktile | [link](matmul_2d_blocktile.cu) | 22.4 T (42.9%) | 11.1 T (59.5%) | 16.0 T (68.7%) | **−4.9 T** ⚠️ |
| **2D blocktile (autotuned)** | [link](matmul_2d_blocktile.cu) | **34.0 T (65.2%)** | **16.4 T (88.0%)** | **19.7 T (84.8%)** | **−3.3 T** |
| Vectorized | [link](matmul_vectorized.cu) | 32.9 T (63.0%) | 13.9 T (74.8%) | 18.2 T (78.4%) | **−4.3 T** ⚠️ |
| **Vectorized (autotuned)** | [link](matmul_vectorized.cu) | **34.8 T (66.7%)** | **16.6 T (89.0%)** | — | — |
| Warptile | [link](matmul_warptile.cu) | 28.3 T (54.2%) | 14.0 T (75.5%) | 21.8 T (93.7%) | **−7.8 T** ⚠️⚠️ |
| **Warptile (autotuned)** | [link](matmul_warptile.cu) | **33.4 T (64.4%)** | **15.0 T (80.7%)** | — | — |

**Baselines**: H100 cuBLAS FP32 = 52.2 T | A100 cuBLAS FP32 = 18.6 T | Simon's A6000 cuBLAS TF32 = 23.2 T

Each `link` in the Src column points to the corresponding `matmul_<step>.cu` file in this directory. Simon's autotuned results are: 2D blocktile = 19.7 T; warptile was hand-tuned (not autotuned).

### What This Table Actually Shows

**Simple kernels (naive → 1D blocktile):** We win on absolute TFLOPS — our A100 has 29% more SMs (108 vs 84), and these kernels are compute-bound rather than SMEM/latency-bound, so more SMs directly translates to more throughput.

**From 2D blocktile onward, Simon's kernels are fundamentally more efficient.** Despite having fewer SMs, his 2D blocktile hits 16.0 T vs our 11.1 T. Per-SM: Simon = 190 GFLOPS/SM vs us = 103 GFLOPS/SM — his kernels extract 84% more per SM at this complexity level.

**Our autotuning helps significantly** (+48% on 2D, from 11.1→16.4 T), but doesn't close the gap. His autotuned 2D blocktile at 19.7 T is still 1.6× per-SM of ours. The gap originates in tile design (register layout, load patterns, loop ordering) rather than parameter selection alone.

**cuBLAS baseline can't be used for cross-paper normalization**: his % is TF32-relative (23.2 T), ours is FP32-relative (18.6 T). TFLOPS is the common unit — by that measure, Simon's absolute 19.7 T > our 16.4 T at 2D autotuned.

### Why Simon's Kernels Are More Efficient Per-SM

Source-code reading of `siboehm/SGEMM_CUDA/src/runner.cu` reveals key design differences:

1. **2D blocktile loads**: Simon uses `float4` GMEM loads with stride decoupling into SMEM, then scalar tensor-core-compatible layout. Our 2D blocktile uses scalar loads. The float4 path reduces GMEM transactions by 4× at the load boundary.

2. **Register pressure management**: Simon's 2D blocktile uses 8×8 thread tiles (\(TM=TN=8\), 64 reg values per thread) and keeps loop ordering that lets the compiler hoist `regA[i]`. Our non-autotuned version used the same TM/TN but with different loop ordering that causes more register pressure.

3. **SMEM address pattern**: Simon's autotuned config (from his kernel 9) avoids bank conflicts through offset addressing. Our early versions had 2-way bank conflicts (fixed in commit `38f8709` via the `As[BM][BK+1]` padding).

4. **Vectorized: He uses float4 loads + shared memory**. Our hardcoded vectorized uses float4 loads but inline into registers. His path achieves better SMEM bandwidth utilization because the SMEM cache is fully utilized across warps.

5. **Warptile: This is Simon's magnum opus.** 21.8 T on 84 SMs = 260 GFLOPS/SM — higher than our cuBLAS FP32 on 108 SMs! His warp-level tiling uses per-warp register files + explicit warp synchronization patterns that are tuned to Ampere's sub-partition layout. This is the only kernel where expertise (knowing the SM microarchitecture) visibly dominates over methodology (sweeping parameters).

### H100 Head-to-Head

| Kernel | TFLOPS | vs cuBLAS FP32 |
|---|---|---|
| Simon's warptile (on H100, Pranjal's data) | 31.8 | 60.9% |
| **Our vectorized (autotuned)** | **34.8** | **66.7%** |
| **Our warptile (autotuned)** | **33.4** | **64.4%** |

On H100, our autotuned kernels edge Simon's warptile (34.8 T > 31.8 T). But this is the only hardware where we win at the top tier — and it's H100-specific autotuning, not algorithmic superiority.

cuBLAS baselines: H100 FP32 = 52.2 TFLOPS, A100 FP32 = 18.6 TFLOPS, Simon's A6000 TF32 = 23.2 TFLOPS.

---

## Table 2: Tensor Core Path — Ours vs Pranjal (H100 Worklog)

**N=4096. Both vs cuBLAS BF16.** Pranjal's baseline = 716.7 TFLOPS, ours = 493.6 TFLOPS.

| Step | Technique | File | Pranjal (4K) | Ours (4K) | Status |
|---|---|---|---|---|---|
| — | Simon's FP32 (H100) | — | 4.4% (31.8 T) | 4.6% (32.9 T) | ✅ Beat Simon |
| K1 | **Tensor Core** | [link](matmul_wmma.cu) | **44.3%** (317.6 T) | **5.7%** (27.5 T) | ⚠️ **7.8× gap** |
| K2 | Larger tiles | 🆕 | 59.0% (423 T) | — | |
| K3 | Async loads (TMA) | 🆕 | 69.5% (498 T) | — | |
| K4 | Pushing tile size limit | 🆕 | 88.2% (632 T) | — | |
| K5 | Hide store latency | 🆕 | 92.1% (660 T) | — | |
| K6 | Faster barriers | 🆕 | 98.4% (705 T) | — | |
| K7 | Thread Block Clusters | 🆕 | **102.4%** (734 T) | — | Surpasses cuBLAS! |
| K8 | Micro-optimizations | 🆕 | 104.3% (747 T) | — | |
| K9 | Async Stores | 🆕 | 105.8% (759 T) | — | |
| K10 | Hilbert Curves | 🆕 | 106.6% (764 T) | — | |

Also available: [`matmul_wmma_bf16.cu`](matmul_wmma_bf16.cu) (WMMA BF16 variant, 27.6T at 4K), [`matmul_cublas_bf16.cu`](matmul_cublas_bf16.cu) (cuBLAS BF16 baseline, 493.6T at 4K).

### The API Gap (Why K1 = 7.8×)

| | Pranjal K1: "Tensor Core" | Our WMMA |
|---|---|---|
| API | **WGMMA** (warp-group, 128 threads) | `nvcuda::wmma` (1 warp, 32 threads) |
| SM sub-partitions used | 4/4 | 1/4 |
| Tile size | 64×M (Hopper-native) | 16×16 (Volta-era) |
| Memory | Shared memory tile cache | Direct global memory reads |
| Nsight profile | Not published | Memory 93%, Compute 17% — TC starving |

Switching from WMMA to WGMMA is expected to close most of the 7.8× gap in a single change. Every subsequent step (TMA, pipelining, clusters) builds on WGMMA.

Note: our WMMA at 4K (5.7%, 27.5T) is better than at 2K (2.6%, 25.6T) — larger matrix = more tiles = higher SM occupancy. But still 7.8× behind WGMMA.

---

## Verification Notes

**Our A100 numbers** verified on 2026-06-01 ~10:25 AM PT via IAP tunnel to `a100-spot-5` (us-east1-b). Fresh rebuild from `main` branch, all kernels in single session. Reproducibility ±0.5% on 100-iteration medians.

**Our H100 numbers** verified on `pi1-h100-27` (job 11723, `--gres=gpu:8 --exclusive`). Run-to-run reproducibility ±0.07 T on auto kernels.

**Simon's numbers** extracted from his README.md (GPUs marker: "NVIDIA A6000 (Ampere)") and `siboehm/SGEMM_CUDA` source. cuBLAS mode confirmed by reading `src/runner.cu` line 129-131.

**Earlier errors corrected** (2026-06-01):
- Simon's GPU was labeled "A100" throughout the doc — corrected to **A6000** after source-code audit
- Gap column previously compared % vs different cuBLAS modes — replaced with absolute TFLOPS
- A100 head-to-head section removed (different hardware)

---

## Summary

| Path | Progress | Key Blocker | Next Step |
|---|---|---|---|
| **FP32** (Simon) | 8/8 done, autotuned | Simon's 2D+ kernels still 1.6× more efficient per-SM | Study his register layout patterns |
| **TC** (Pranjal) | 0/10 done, WMMA at 5.7% | WMMA API (need WGMMA) | **WGMMA** — step 1 of 10 |

---

## Action Items

| Priority | What | Expected Gain |
|---|---|---|
| **P0** | WGMMA — Pranjal K1 | 5.7% → ~44% (7.8×) |
| P2 | Study Simon's 2D tile register layout for per-SM efficiency | |
| P2 | TMA (async copy) — Pranjal K3 | 44% → 70% |
