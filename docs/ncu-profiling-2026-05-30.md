# Nsight Compute Profiling — Matmul 1024×1024, H100 SM90

**Date**: 2026-05-30  
**Node**: pi1-h100-11 (Job 11711)  
**GPU**: NVIDIA H100 80GB HBM3, 132 SMs @ 1.98 GHz boost, CUDA 12.8  
**Profile tool**: ncu 2025.1.1.0  
**Binary**: `~/matmul_profile` (11 methods, single launch each, N=1024, block dim=16)

## Speed of Light Throughput

| Metric | Naive | Vectorized (f4) | Warptile | cuBLAS (TF32) | WMMA | WMMA BF16 | cuBLAS BF16 |
|---|---|---|---|---|---|---|---|
| **Duration (µs)** | 513.4 | 240.9 | — | 70.4 | 101.4 | 101.7 | **9.0** |
| **Compute (SM) %** | 61.8 | 18.8 | — | 70.2 | 16.5 | 16.4 | 30.2 |
| **Memory %** | 92.8 | 24.9 | — | 61.1 | 93.2 | 93.0 | 32.9 |
| **DRAM %** | 0.5 | 1.0 | — | 3.6 | 1.2 | 1.2 | **14.0** |
| **L1/TEX %** | 96.2 | 51.9 | — | 64.9 | 98.3 | 98.4 | 46.9 |
| **L2 %** | 17.6 | 3.3 | — | 20.6 | 15.2 | 14.9 | 33.7 |
| **SM Freq (GHz)** | 1.60 | 1.60 | — | 1.59 | 1.44 | 1.44 | 1.46 |
| **DRAM Freq (GHz)** | 2.62 | 2.62 | — | 2.62 | 2.62 | 2.62 | 2.61 |
| **Elapsed Cycles** | 823,870 | 386,578 | — | 112,464 | 146,158 | 146,459 | 13,256 |

## Kernel Launch Mapping

| Launch Index | Kernel Name | Method |
|---|---|---|
| 0 | `matmulNaiveKernel` | Naive |
| 1 | `matmulCoalescedKernel` | Coalesced |
| 2 | `matmulSmemKernel` | Shared memory |
| 3 | `matmul1DBlocktileKernel` | 1D Blocktile |
| 4 | `matmul2DBlocktileKernel` | 2D Blocktile |
| 5 | `matmulVectorizedKernel` | Vectorized float4 |
| 6 | `matmulWarptileKernel` | Warptile |
| 7 | `sm80_xmma_gemm_f32f32_f32f32_...` | cuBLAS (TF32 TC) |
| 8-9 | `convertFP32ToFP16` | (WMMA prep) |
| 10 | `matmulWMMAKernel` | WMMA |
| 11-12 | `convertFP32ToBF16Kernel` | (WMMA BF16 prep) |
| 13 | `matmulWmmaBf16Kernel` | WMMA BF16 |
| 14-15 | `convertFP32ToBF16` | (cuBLAS BF16 prep) |
| 16 | `nvjet_tss_128x64_64x8_1x2_h_b...` | cuBLAS BF16 |

## Diagnosis

### Naive → Memory-bound
Memory 92.8%, L1 96.2%, DRAM 0.5%. The kernel is L1-thrashing — B matrix strided reads cause constant L1 replay. Compute 61.8% is misleading: those are stalls disguised as active cycles.

### Vectorized (hand-written best) → Latency-bound
Memory dropped from 92.8% → 24.9% (float4 reduced transaction count, good). But Compute only 18.8% — the SM is idle not because data is slow, but because occupancy is too low. Likely register pressure or suboptimal block size for this N.

### WMMA / WMMA BF16 → Memory-bound
Memory >93%, Compute <17%. At N=1024, the WMMA tile (16×16) is too small to saturate Tensor Cores. The SM has 4 sub-partitions but WMMA only uses 1 warp (1 sub-partition). The other 3 sub-partitions are idle. Need WGMMA (128 threads) on H100.

### cuBLAS (TF32) → Still memory-bound
Compute 70.2%, Memory 61.1%. Approaching the roofline, but memory throughput is the limiting ceiling.

### cuBLAS BF16 (`nvjet_tss`) → Most balanced
Compute 30.2%, DRAM 14%, L2 33.7%. NVIDIA's proprietary kernel keeps data in L2 cache — 9 µs is 57× faster than naive. The `nvjet_tss` kernel uses aggressive tiling + L2 residency control that our hand-written kernels lack.

## Performance Benchmarks (from `matmul_test -m all`)

Full sweep 11 methods × 6 sizes. N=512, 1024, 2048 highlighted.

| Method | N=512 | N=1024 | N=2048 |
|---|---|---|---|
| naive | 4.7 TFLOPS (1.0%) | SKIPPED | SKIPPED |
| coalesced | 5.9 TFLOPS (1.2%) | 6.5 TFLOPS (1.3%) | 6.6 TFLOPS (1.3%) |
| smem | 8.1 TFLOPS (1.6%) | 9.0 TFLOPS (1.8%) | 9.2 TFLOPS (1.9%) |
| 1d_blocktile | 6.6 TFLOPS (1.3%) | 16.4 TFLOPS (3.3%) | 16.9 TFLOPS (3.4%) |
| 2d_blocktile | 2.8 TFLOPS (0.6%) | 12.6 TFLOPS (2.5%) | 21.6 TFLOPS (4.4%) |
| **vectorized** | 3.1 TFLOPS (0.6%) | 12.7 TFLOPS (2.6%) | **32.8 TFLOPS (6.6%)** |
| warptile | 2.1 TFLOPS (0.4%) | 7.8 TFLOPS (1.6%) | 28.4 TFLOPS (5.7%) |
| cuBLAS (TF32) | 18.8 TFLOPS (3.8%) | 38.3 TFLOPS (7.7%) | 50.4 TFLOPS (10.2%) |
| WMMA | 8.5 TFLOPS (1.7%) | 22.5 TFLOPS (4.5%) | 25.6 TFLOPS (5.2%) |
| WMMA BF16 | 8.6 TFLOPS (1.7%) | 22.9 TFLOPS (4.6%) | 25.6 TFLOPS (5.2%) |
| **cuBLAS BF16** | 27.4 TFLOPS (5.5%) | 136.1 TFLOPS (27.5%) | **287.1 TFLOPS (58.0%)** |

MFU denominator: 495 TFLOPS (TF32 Tensor Core peak) — same denominator for all methods, so FP32 hand-written kernels are penalized ~7.4× relative to their actual FP32 peak (67 TFLOPS).

## Key Takeaway

Our hand-written kernels cap at ~33 TFLOPS (vectorized float4 @ N=2048). cuBLAS BF16 hits 287 TFLOPS at the same size — **8.7× gap**. The path to close it: WGMMA (full SM utilization) → TMA (async copy) → pipelining. The existing WMMA kernel uses only 1 warp out of 4 per SM — this is the single biggest fix.

## Raw ncu-rep Files

- `~/matmul_ncu_0.ncu-rep` through `~/matmul_ncu_16.ncu-rep` on pi1-h100-11 home (NFS)
- Each file = 1 kernel launch, 10 passes
- Extract: `sudo /usr/local/cuda/bin/ncu -i <file>.ncu-rep --print-summary per-kernel`
