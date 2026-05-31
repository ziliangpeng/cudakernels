# Matmul Optimization Worklog

**Date**: 2026-05-30  
**GPU**: H100 80GB HBM3, SM90, 132 SMs @ 1.98 GHz  
**Baseline**: cuBLAS FP32 (`CUBLAS_PEDANTIC_MATH`) — 52.2 TFLOPS @ 4K  
**Node**: pi1-h100-11  

Each kernel = one class (`Matmul*`) in `matmul_*.{h,cu}`. Benchmark harness in `matmul.cpp`. All numbers at N=4096 unless noted.

---

## Step 1: Naive

**File**: [`matmul_naive.cu`](matmul_naive.cu) | **Class**: `MatmulNaive`

### What it does
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

### Fatal flaw
16×16 block dim → one warp (32 threads) spans two rows. `threadIdx.x` wraps at 16, so first 16 threads = row R, next 16 = row R+1. 
- A load: two transactions (half-warp each, addresses N×4 bytes apart)
- B load: two transactions reading **the same 16 addresses twice** (redundant work)

### Why 16×16?
Safest default — 256 threads works on any GPU. But it splits warps across two rows, wasting half of every memory transaction.

### Performance

| Block Dim | Threads | TFLOPS | % vs cuBLAS FP32 (52.2T) |
|---|---|---|---|
| 16×16 | 256 | 5.3 T | 10.2% |
| 32×32 | 1024 | 6.1 T | 11.8% |

32×32 closes the warp-splitting problem (32 consecutive threads all share the same row) but hits the 1024-thread block limit, hurting occupancy.

### What we learned
Warp-to-row mapping matters. 32 threads per warp should always compute elements in the same row when possible. 16×16 works against this, 32×32 or 1D mapping works with it.

---

## Step 2: Coalesced

**File**: [`matmul_coalesced.cu`](matmul_coalesced.cu) | **Class**: `MatmulCoalesced`

### What it does
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

### The fix
Uses 1D block indexing + manual division to guarantee 32 consecutive threads map to the same row.

- A load: all 32 threads read the same `A[row][k]` → **hardware broadcast**, single transaction
- B load: 32 consecutive columns → **single 128-byte coalesced transaction**

Same insight as 32×32 naive, but encodes the mapping in code rather than relying on block shape.

### Performance

| Kernel | TFLOPS | % vs cuBLAS FP32 | Improvement |
|---|---|---|---|
| Naive (16×16) | 5.3 T | 10.2% | — |
| **Coalesced** | **5.7 T** | **10.9%** | +7% |

### What we learned
Coalescing fixes the immediate waste but doesn't reduce total memory accesses. Every thread still reads every element of A and B from GMEM, N times. No cross-thread or cross-k reuse. That's the next step.

---

## Step 3: SMEM Tiling

**File**: [`matmul_smem.cu`](matmul_smem.cu) | **Class**: `MatmulSmem`

### What it does
```
Each block loads 32×32 tiles into shared memory, then computes from on-chip SRAM
→ GMEM reads reduced by 32× (one load per tile per thread, reused across k-loop)
```
```c
__shared__ float As[32][32], Bs[32][32];  // Tile cache per block

for (int tileIdx = 0; tileIdx < N; tileIdx += 32) {
    // 1. Cooperative tile load: 1024 threads each load 1 element
    As[ty][tx] = A[row * N + (tileIdx + tx)];
    Bs[ty][tx] = B[(tileIdx + ty) * N + col];
    __syncthreads();

    // 2. Compute from SMEM: 32 madd per thread, all on-chip (~19 TB/s)
    #pragma unroll
    for (int k = 0; k < 32; k++)
        sum += As[ty][k] * Bs[k][tx];
    __syncthreads();
}
```

### Memory access reduction
```
Naive:  each thread reads 2 × N floats from GMEM = 4K × 2 = 8K reads/thread
SMEM:   each thread reads 2 floats per tile × (N/32 tiles) = 4K/16 = 256 reads/thread
        → 32× fewer GMEM reads
```

The K-loop inner body reads entirely from shared memory (~19 TB/s bandwidth) instead of HBM (~3 TB/s).

### Thread mapping
```
32×32 2D block = 1024 threads
Thread (tx, ty) loads As[ty][tx] and Bs[ty][tx] from GMEM
Then computes As[ty][k] × Bs[k][tx] for k=0..31
```
Each thread still computes exactly one output element — the tiling is at the block level, not per-thread.

### Bank conflict analysis
- `As[ty][k]`: ty varies across threads (0..31), k fixed → stride 32 → different banks → **no conflict**
- `Bs[k][tx]`: tx varies, k fixed → stride 1 → contiguous → **no conflict**

### Performance

| Kernel | TFLOPS | % vs cuBLAS FP32 | Improvement |
|---|---|---|---|
| Coalesced | 5.7 T | 10.9% | — |
| **SMEM tiling** | **9.0 T** | **17.2%** | +59% |

### What we learned
Shared memory is the single biggest jump for memory-bound kernels. 32× reduction in GMEM traffic translates directly to throughput. But each thread still computes only 1 output element — next step is to have each thread compute more.

---

## To be continued...

TODO: 1D blocktile, 2D blocktile, vectorized, warptile — document each step's insight.
