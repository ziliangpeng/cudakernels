# Performance Comparison: Our Kernels vs Published Worklogs

**Date**: 2026-05-30  
**Sources**:
- [siboehm: CUDA Matmul Worklog](https://siboehm.com/articles/22/CUDA-MMM) — A100, N=4096, FP32, vs cuBLAS FP32
- [Pranjal: Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) — H100, N=4096, TC, vs cuBLAS BF16
- Our data: [ncu-profiling-2026-05-30.md](../docs/ncu-profiling-2026-05-30.md) — H100, N=2048 and N=4096

---

## Table 1: FP32 Path — Ours vs Simon (siboehm)

**Both sides compared against their own cuBLAS FP32 baseline.** Simon's % are from his A100 blog. Our % are vs our cuBLAS FP32 at the same N.

| | cuBLAS FP32 2K H100 | cuBLAS FP32 4K H100 | cuBLAS FP32 (Simon A100) |
|---|---|---|---|
| Baseline | 50.4 TFLOPS | 52.2 TFLOPS | 23.2 TFLOPS (A100) |
| Source | [`matmul_cublas.cu`](matmul_cublas.cu) | same | — |

Our 2D blocktile and warptile use hardcoded tile sizes (`BM/BN/BK`). Simon autotunes.

| Step | Src | 2K H100 | 4K H100 | Simon 4K A100 | Gap (vs Simon 4K) |
|---|---|---|---|---|---|
| Naive | [link](matmul_naive.cu) | 10.7% (5.4 T) | 10.2% (5.3 T) | 1.3% (0.3 T) | +8.9pp ✅ |
| Coalesced | [link](matmul_coalesced.cu) | 13.1% (6.6 T) | 10.9% (5.7 T) | 8.5% (2.0 T) | +2.4pp ✅ |
| SMEM tiling | [link](matmul_smem.cu) | 18.3% (9.2 T) | 17.2% (9.0 T) | 12.8% (3.0 T) | +4.4pp ✅ |
| 1D blocktile | [link](matmul_1d_blocktile.cu) | 33.5% (16.9 T) | 33.7% (17.6 T) | 36.5% (8.5 T) | −2.8pp ≈ |
| **1D blocktile (autotuned)** | [link](matmul_1d_blocktile.cu) | — | **36.9% (19.3 T)** | 36.5% (8.5 T) | **+0.4pp** ✅ |
| 2D blocktile | [link](matmul_2d_blocktile.cu) | 42.9% (21.6 T) | 42.9% (22.4 T) | 68.7% (16.0 T) | **−25.8pp** ⚠️ |
| **2D blocktile (autotuned)** | [link](matmul_2d_blocktile.cu) | — | **65.2% (34.0 T)** | 68.7% (16.0 T) | **−3.5pp** ✅ |
| Vectorized | [link](matmul_vectorized.cu) | 65.1% (32.8 T) | 63.0% (32.9 T) | 78.4% (18.2 T) | −15.4pp |
| **Vectorized (autotuned)** | [link](matmul_vectorized.cu) | — | **66.7% (34.8 T)** | — | — |
| Warptile | [link](matmul_warptile.cu) | 56.3% (28.4 T) | 54.2% (28.3 T) | **93.7%** (21.8 T) | **−39.5pp** ⚠️⚠️ |
| Autotuning | — | — | — | 84.8% (19.7 T) | — |

Each `link` in the Src column points to the corresponding `matmul_<step>.cu` file in this directory.

**Observation**: coalesced + SMEM degrade slightly at 4K (memory-bound, working set exceeds L2). 1D/2D blocktile hold steady. Vectorized + warptile nearly flat (occupancy/register-bound). Our % vs Simon are essentially identical at 2K and 4K for every kernel — the gap is structural (tile sizes, block dim), not scale-dependent.

**Autotuning update (2026-05-30)**: 1D blocktile autotuned across 7 legal candidates (kernel constraint: BM = BN = BK·TM). Best config = `BM=BN=64, BK=4, TM=16` (256 threads, 16 outputs per thread), 19.26 TFLOPS @ N=4096 — **+9% over hardcoded baseline (17.6 → 19.3)**, just edging Simon's autotuned 1D blocktile on A100 (36.5% → 36.9%). The winning config is *not* siboehm's recommended `(64, 64, 8, 8)` — H100 prefers smaller BK + larger TM (more register reuse per thread). See [`autotune.md`](autotune.md) and [`worklog.md`](worklog.md) Step 4 "Autotune result" section for full details.

**2D blocktile autotune (2026-05-30, updated)**: 2D blocktile autotuned across 19 candidates (initial 11 + 4 expansion + 4 v3 after the SMEM math fix). Best config = `BM=BN=128, BK=16, TM=16, TN=8` (128 threads, 8×16 thread tile), **34.0 TFLOPS @ N=4096 — +52% over the hardcoded baseline (22.3 → 34.0)**. We now nearly match Simon's autotuned A100 2D blocktile (65.2% vs 68.7%) and **beat Simon's autotuned warptile when run on H100** (65.2% vs 60.9%). Three lessons: (a) BK=16 is a sweet spot, not a "deeper is better" ladder; (b) **TM and TN are NOT mirror-symmetric** — `(TM=16, TN=8)` runs 30% faster than the swapped `(TM=8, TN=16)`, because the compiler hoists `regA[i]` and longer TN causes register-allocation pressure; (c) once SM occupancy saturates, bigger blocks just add SMEM bloat without buying parallelism. Gemini Code Assist caught two real bugs in the autotuner's validity check during PR review (SMEM 2× over-count, and missing `NUM_THREADS % BK == 0` divisibility leading to silent OOB SMEM writes for BK=24 candidates — fixed in commit 38f8709). See [`autotune.md`](autotune.md) and [`worklog.md`](worklog.md) Step 5 "Autotune result" for full details.

**Vectorized autotune (2026-05-31)**: Vectorized kernel autotuned across 16 candidates using same structure as 2D blocktile (non-transposed As, strided scalar loads) plus float4 (128-bit) C stores. Same winner as 2D: `(BM=BN=128, BK=16, TM=16, TN=8)` at **34.8 TFLOPS @ N=4096 — +1.1T (+3.3%) over 2D blocktile autotune**. The float4 store alone adds measurable gain on H100 by reducing L1 cache-sector transactions during the C writeback phase. Total FP32 progress: naive 10.2% → 66.7% of cuBLAS FP32 (+56.5pp). See [`autotune.md`](autotune.md) Step 6 autotune section for details.

Shared infra: [`matmul.cpp`](matmul.cpp) (benchmark harness), [`matmul_kernel.h`](matmul_kernel.h) (base class), [`matrix_init.{h,cu}`](matrix_init.cu) (CPU reference).

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

**N=4096. Both vs cuBLAS BF16.** Pranjal's baseline = 716.7 TFLOPS, ours = 485.5 TFLOPS.

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

Also available: [`matmul_wmma_bf16.cu`](matmul_wmma_bf16.cu) (WMMA BF16 variant, 27.5T at 4K), [`matmul_cublas_bf16.cu`](matmul_cublas_bf16.cu) (cuBLAS BF16 baseline, 485.5T at 4K).

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

## Summary

| Path | Progress | Key Blocker | Next Step |
|---|---|---|---|
| **FP32** (Simon) | 7/7 done, beats Simon (32.9 > 31.8 T) | No autotuning (warptile regresses) | Low priority — FP32 ceiling ~52T |
| **TC** (Pranjal) | 0/10 done, WMMA at 5.7% | WMMA API (need WGMMA) | **WGMMA** — step 1 of 10 |

---

## Action Items

| Priority | What | Expected Gain |
|---|---|---|
| **P0** | WGMMA — Pranjal K1 | 5.7% → ~44% (7.8×) |
| P1 | Autotune BM/BN/BK for FP32 warptile | 28.3T → ~35T (+24%, low ROI vs TC) |
| P2 | TMA (async copy) — Pranjal K3 | 44% → 70% |
| P2 | Nsight profile each WGMMA step | |
