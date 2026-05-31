# Matmul Optimization Worklog

**Date**: 2026-05-30  
**GPU**: H100 80GB HBM3, SM90, 132 SMs @ 1.98 GHz  
**Baseline**: cuBLAS FP32 (`CUBLAS_PEDANTIC_MATH`) — 50.4 TFLOPS @ 2K, 52.2 TFLOPS @ 4K  
**Node**: pi1-h100-11  

Each kernel = one class (`Matmul*`) in `matmul_*.{h,cu}`. Benchmark harness in `matmul.cpp`.

---

## Step 1: Naive

**File**: [`matmul_naive.cu`](matmul_naive.cu)

**What it does**:
```
Each thread computes C[row][col] = dot(A[row], B[col])
→ 256 threads (16×16 2D block), one output element per thread
```
```c
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
for (int k = 0; k < N; k++)
    sum += A[row * N + k] * B[k * N + col];
```

**Fatal flaw**: 16×16 block dim → one warp (32 threads) spans two rows. `threadIdx.x` wraps at 16, so first 16 threads = row R, next 16 = row R+1. A load requires two transactions (half-warp each), B load reads the same addresses twice (duplicate work).

**Why 16×16?** Safest default — 256 threads works everywhere. But it splits warps across two rows, wasting half of every memory transaction.

**Performance**:

| Block Dim | Threads | 4K TFLOPS | % vs cuBLAS FP32 |
|---|---|---|---|
| 16×16 | 256 | 5.3 | 10.2% |
| 32×32 | 1024 | 6.1 | 11.8% |

32×32 closes the warp-splitting problem (32 consecutive threads all share the same row) but hits the 1024-thread block limit, hurting occupancy.

---

## Step 2: Coalesced

**File**: [`matmul_coalesced.cu`](matmul_coalesced.cu)

**What it does**:
```
Explicit row/col mapping: threadCol = threadIdx.x % 32, threadRow = threadIdx.x / 32
→ 1024 threads per block (1D), same row for all 32 threads in a warp
```
```c
int threadCol = threadIdx.x % 32;
int threadRow = threadIdx.x / 32;
int row = blockRow * 32 + threadRow;
int col = blockCol * 32 + threadCol;
const float *A_row = A + row * N;
const float *B_col = B + col;
for (int k = 0; k < N; k++)
    sum += A_row[k] * B_col[k * N];
```

**The fix**: Uses 1D block indexing + manual division to guarantee 32 consecutive threads map to the same row. 

- A load: all 32 threads read the same `A[row][k]` → **hardware broadcast**, single transaction
- B load: 32 consecutive columns → **single 128-byte coalesced transaction**

**Same idea as 32×32 naive**, but doesn't need to change blockDim — it's done in the kernel code itself.

**Performance**:

| | 2K H100 | 4K H100 | Note |
|---|---|---|---|
| Naive (16×16) | 10.7% (5.4T) | 10.2% (5.3T) | |
| **Coalesced** | **13.1%** (6.6T) | **10.9%** (5.7T) | +2.4pp @ 2K, +0.7pp @ 4K |

Coalescing helps but doesn't reduce the total number of memory accesses — every thread still reads every element of A and B from global memory, N times. No cross-thread or cross-k reuse. That's what shared memory tiling addresses next.

---

## To be continued...

TODO: SMEM, 1D blocktile, 2D blocktile, vectorized, warptile — document each step's insight.
