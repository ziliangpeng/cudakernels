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

---

## Step 4: 1D Blocktile — Thread-Level Reuse via Registers

**File**: [`matmul_1d_blocktile.cu`](matmul_1d_blocktile.cu) | **Class**: `Matmul1DBlocktile`

### Core insight: two levels of reuse

SMEM tiling (Step 3) solved **block-level reuse**: 1024 threads cooperatively load a tile from HBM into shared memory once, then everyone reads from on-chip SRAM instead of HBM. This exploits the L1/SMEM SRAM as a bandwidth amplifier.

**But** inside each thread, the k-loop still Read → Use Once → Discard:

```c
// SMEM kernel: thread (ty=3, tx=5) — one output, bandwidth-inefficient within thread
for (int k = 0; k < 32; k++)
    sum += As[3][k] * Bs[k][5];  // read Bs[k][5] from SMEM, use once, throw away
```

Every SMEM read produces exactly **1 madd**. SMEM bandwidth is ~19 TB/s per SM — fast, but only half-utilized when reads are 1:1 with computation.

1D Blocktile adds **thread-level reuse via registers**: each thread computes 8 output elements along a column, and reuses each `B` value from SMEM across all 8 partial sums stored in registers.

### Memory hierarchy: where each level's reuse happens

```
┌──────────────────────────────────────────────────────────────┐
│  HBM (80GB, 3.35 TB/s)                                        │
│  Naive/Coalesced: each thread reads from here, N times        │
│  No reuse — everyone independently reloads the same data      │
└────────────────────┬─────────────────────────────────────────┘
                     │ SMEM tiling: cooperative tile load
                     │ 32× fewer HBM trips per thread
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  SMEM / L1 (256KB per SM, ~19 TB/s)                           │
│  SMEM tiling: block-level reuse                               │
│    → Read A/B tile into shared memory once                    │
│    → All threads in block share the tile                      │
│  ⚠️ But each thread: read → use once → discard → read again   │
│    → 1 SMEM read = 1 madd (arithmetic intensity too low)      │
└────────────────────┬─────────────────────────────────────────┘
                     │ 1D Blocktile: store B value in register
                     │ Reuse it across 8 output elements
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Register File (256KB per SM, ~0 cycle latency)               │
│  1D Blocktile: thread-level reuse                             │
│    → Load Bs[dotIdx][threadCol] into register (tmpB)          │
│    → Multiply against 8 different A rows stored in registers  │
│    → 1 SMEM read = 8 madds (8× arithmetic intensity)          │
│    → Equivalent SMEM bandwidth amplified 8×                   │
└──────────────────────────────────────────────────────────────┘
```

| Level | What is reused | Who reuses it | Mechanism |
|---|---|---|---|
| SMEM / L1 | A and B tiles | All threads in the block | `__shared__` scratchpad |
| Register File | B element (`tmpB`) | A single thread, across 8 partial sums | Local variable held in register |

### What the code does

```
BM=64, BN=64, BK=8, TM=8
512 threads per block (64×64/8 = 512 instead of 1024)

Each thread:
  threadCol (0-63): which column in the output block
  threadRow (0-7):  which group of 8 rows (threadRow * 8 through threadRow * 8 + 7)

  threadResults[TM] = {0, 0, 0, 0, 0, 0, 0, 0};  // 8 partial sums in registers
```

```c
// Key inner loop — tmpB reused 8 times inside each dotIdx iteration
for (int dotIdx = 0; dotIdx < BK_1D; dotIdx++) {      // BK=8
    float tmpB = Bs[dotIdx][threadCol];                // ← 1 SMEM read
    for (int resIdx = 0; resIdx < TM_1D; resIdx++) {   // TM=8
        threadResults[resIdx] +=                       // ← 8 register accumulators
            As[threadRow * TM + resIdx][dotIdx] * tmpB; //   all reuse the same tmpB
    }
}
```

The critical line is `As[threadRow * TM + resIdx][dotIdx]` — as `resIdx` varies from 0 to 7, we stride across 8 consecutive rows of `As`, each contributing one `A` value per iteration. `tmpB` stays in a register across all 8 madd operations.

### Performance

| Kernel | TFLOPS | % vs cuBLAS FP32 | Improvement |
|---|---|---|---|
| SMEM tiling | 9.0 T | 17.2% | — |
| **1D blocktile** | **17.6 T** | **33.7%** | **+96%** |

### What we learned

> **SMEM tiling reuses data at the L1/SMEM level (block-level reuse). 1D blocktile adds reuse at the register level (thread-level reuse).**

The 96% jump (from 5.7T to 9.0T was block-level SMEM; from 9.0T to 17.6T was register-level thread reuse) comes from making each SMEM read do 8× more work. The same B element is read once from SMEM, held in a register, and multiplied against 8 different A values — each producing a different partial sum. This amplifies the effective SMEM bandwidth by 8× without changing the tile loading pattern at all.

The memory hierarchy insight: HBM → SMEM/L1 fixed the *cross-thread* waste; Register reuse fixed the *within-thread* waste. Both layers are needed to approach the hardware limit.
