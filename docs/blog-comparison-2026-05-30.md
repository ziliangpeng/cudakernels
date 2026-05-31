# Performance Comparison: Our Kernels vs Published Worklogs

**Date**: 2026-05-30  
**Sources**:
- [siboehm: CUDA Matmul Worklog](https://siboehm.com/articles/22/CUDA-MMM) — A100, N=4096, FP32 path, vs cuBLAS FP32
- [Pranjal: Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) — H100, N=4096, TC path, vs cuBLAS BF16
- Our data: [ncu-profiling-2026-05-30.md](ncu-profiling-2026-05-30.md) — H100, N=1024/2048/4096, both paths

> **Caveat**: Matrix sizes differ (our N=1024/2048 vs their N=4096 for step-by-step), and % values use each blog's own cuBLAS baseline. FP32 baseline is FP32 cuBLAS for siboehm, TF32 TC cuBLAS (495T) for us.

## FP32 Hand-Written Kernel Path

Pranjal took siboehm's final warptile kernel and ran it on H100: **31.8 TFLOPS, 4% of cuBLAS TF32 TC** (717T). This is the only FP32 data point from Pranjal's blog — he skips the intermediate FP32 steps and jumps straight to Tensor Cores.

| Step | siboehm (A100, 4K) | Pranjal (H100, 4K) | Ours (H100, 2K) | Notes |
|---|---|---|---|---|
| Naive | 1.3% (309 GFLOPs) | — | — (skipped at 2K) | |
| Coalesced | 8.5% (1,987 G) | — | ~13% (6.6 TFLOPS) | H100 faster GMEM hides coalescing gap |
| SMEM tiling | 12.8% (2,980 G) | — | ~18% (9.2 TFLOPS) | H100's bigger SMEM helps early steps |
| 1D blocktile | 36.5% (8,475 G) | — | ~34% (16.9 TFLOPS) | Nearly identical |
| 2D blocktile | 68.7% (15,972 G) | — | ~43% (21.6 TFLOPS) | **-25.7pp gap** ⚠️ |
| Vectorized | 78.4% (18,237 G) | — | ~65% (32.8 TFLOPS) | **-13.4pp gap** |
| Warptile (Simon final) | 93.7% (21,779 G) | **31.8 TFLOPS** (4% vs cuBLAS TF32) | ~56% (28.4 TFLOPS) | **-37.7pp gap** ⚠️⚠️ |
| Autotuning | 84.8% (19,721 G) | — | — | |

### Diagnosis

Our kernels diverge from siboehm at 2D blocktile. Root causes:

1. **Block dim**: ours = 16×16 fixed for all methods. siboehm = 128×8 for warp tiling and tunes it per step. 16×16 gives only 256 threads — not enough warps to hide latency on H100's 132 SMs.
2. **Matrix size**: N=2048 vs N=4096. At smaller N, tile count / SM is lower, so occupancy drops. H100 has 132 SMs vs A100's 108 — more SMs = even more pressure to have enough tiles.
3. **No autotuning**: siboehm sweeps BM/BN/BK per kernel. Our fixed 16×16 works for naive but hurts advanced kernels.

**Bottom line**: The FP32 optimization path is correct. With proper block dim + autotuning we'd likely match or exceed siboehm's numbers (H100 has faster SM clock + more SMEM per SM).

---

## Tensor Core Path

| Step | Pranjal (H100, 4K) | Ours (H100, 2K) | Notes |
|---|---|---|---|
| Simon's FP32 | 4% (31.8 TFLOPS) | 4.6% (32.9 TFLOPS @ 4K) | We beat Simon on H100 |
| **Tensor Core (WMMA/WGMMA)** | **44%** (317.6 TFLOPS) | **2.6%** (25.6 TFLOPS) | **12.4× gap** ⚠️⚠️ |
| Larger tiles | 59% (423 T) | — | |
| Async loads (TMA) | 70% (498.2 T) | — | |
| Pushing tile limit | 88% (631.9 T) | — | |
| Hide store latency | 92% (660.1 T) | — | |
| Faster barriers | 98% (704.9 T) | — | |
| Thread Block Clusters | 102% (734.2 T) | — | |
| Micro-optimizations | 104% (747.3 T) | — | |
| Async Stores | 106% (758.5 T) | — | |
| Hilbert Curves | 107% (763.9 T) | — | |
| cuBLAS baseline | 100% (716.7 T) | 100% (287 T, BF16 at 2K) | |

### Diagnosis

**12.4× gap at the first TC step.** This is not about algorithmic quality — it's about API choice:

| | Pranjal's first TC kernel | Our WMMA kernel |
|---|---|---|
| API | **WGMMA** (warp-group, 128 threads) | WMMA (single warp, 32 threads) |
| SM sub-partitions used | 4/4 | 1/4 |
| Tile size | 64×M (Hopper-native) | 16×16×16 (Volta-era) |
| Block dim | Optimized for TC occupancy | Hardcoded 16×16 |
| Memory | Shared memory for tile cache | Direct global memory reads |
| Nsight profile | Not available | Memory 93%, Compute 17% — Tensor Core starving |

Pranjal's first TC kernel is already Hopper-optimized (WGMMA). Ours is using Volta's `nvcuda::wmma` API on H100 hardware — only 1 warp out of 4 gets used per SM sub-partition. The nsight data confirms this: Memory 93% means Tensor Cores are idle, waiting for data that a single warp can't supply fast enough.

---

## Simon's Final Algorithm: Head-to-Head (N=4096, H100)

Pranjal reports that Simon's final warptile kernel achieves **31.8 TFLOPS** when run on H100. We benchmarked our best FP32 kernels at the same N=4096:

| Kernel | TFLOPS | MFU (FP32 67T peak) | vs Simon |
|---|---|---|---|
| Simon's warptile (on H100) | 31.8 | 47.5% | — |
| **Our vectorized (float4)** | **32.9** | **49.1%** | **+3.5%** ✅ |
| Our warptile | 28.4 | 42.4% | -10.7% |
| cuBLAS TF32 | 52.3 | 10.6% (TF32 495T) | — |

**We have a kernel that beats Simon's final algorithm** — 32.9 vs 31.8 TFLOPS. The FP32 optimization path (naive → coalesced → smem → 1D/2D blocktile → vectorized) is complete and correct.

However, the FP32 ceiling is ~33 TFLOPS on H100 — that's only **4.6%** of what cuBLAS achieves with Tensor Cores (717T BF16). The FP32 path is a dead end for further gains. The only path forward is the Tensor Core path (WGMMA → TMA → pipelining).

---

## Action Items

| Priority | What | Expected Gain |
|---|---|---|
| **P0** | Switch to WGMMA (warp-group MMA) — Pranjal's first TC step | 2.6% → ~44% MFU (12×+) |
| P1 | Fix block dim: 16×16 → dynamic (sweep or autotune) for FP32 kernels | — (already past Simon) |
| P2 | Add TMA (async copy) — Pranjal step 5 | 44% → 70% |
| P2 | Profile Nsight at each step of the WGMMA path | |
