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

### Recursive tiling — "block tile" is literally a tile of a tile

The single most important conceptual shift at this step:

> **A "block tile" is a tile of a tile.** First we tile the global matrix into block tiles that fit in SMEM. Then we tile each block tile into thread tiles that fit in registers. Each level of tiling unlocks reuse at the next level of the memory hierarchy.

```
Matrix C (4096 × 4096)
  ↓ divided into block tiles (one block per tile)
Block tile (64 × 64)               ← SMEM, shared by all threads in the block
  ↓ divided into thread tiles (one thread per tile)
Thread tile (8 × 1 in 1D blocktile) ← Registers, private to one thread
  ↓ each madd touches
Single elements                     ← register-to-register multiply-add
```

| Tile level | Data lives in | Who shares it | What it eliminates |
|---|---|---|---|
| Block tile (BM × BN) | SMEM | All threads in the block | Cross-thread HBM redundancy |
| Thread tile (TM × 1) | Register | One thread | Repeated SMEM→register reloads inside a thread |

Without the second level of tiling, every madd inside a thread requires a fresh SMEM read for both A and B. The SMEM bandwidth is fine, but the **SMEM ports** become the bottleneck — there are only so many concurrent SMEM accesses per cycle. Once you can't issue enough SMEM reads to feed the FP32 cores, performance plateaus.

Thread-level tiling fixes this by loading a small slice once into registers and reusing it across multiple madds. Each SMEM read now feeds TM=8 madds instead of 1, so the SMEM port traffic drops 8× while compute throughput stays the same.

### "1D" vs "2D" thread tile

- **1D blocktile**: thread tile is TM × 1 (one column slice). One A column held in registers, B element streamed through `tmpB`. 1 SMEM B read → 8 madds.
- **2D blocktile** (next step): thread tile is TM × TN (a small square). Both A column and B row held in registers, mini-square accumulated. 1 SMEM A read + 1 SMEM B read → TM×TN = 64 madds.

The 2D version reuses both A and B at the register level, not just B. That's why it gives another big jump.

### Why not load the whole tile into registers?

A block tile is 64×8 = 512 floats = 2KB. With 512 threads/block and 256KB register file per SM (= 64K 32-bit registers), each thread averages ~128 registers. A single thread cannot hold an entire 512-float tile.

More importantly, registers are **per-thread private**. Loading the whole tile into every thread's registers would be massively redundant — each thread only computes its own 8 output elements and doesn't need the rest of the tile's data in its own registers. The right design is: SMEM holds the tile once (shared), each thread pulls in only its own thread-tile slice.

### What we learned

> **SMEM tiling reuses data at the L1/SMEM level (block-level reuse).
> 1D blocktile adds reuse at the register level (thread-level reuse).
> A "block tile" is a tile of a tile — recursive tiling matches recursive levels of the memory hierarchy.**

The 96% jump comes from making each SMEM read do 8× more work. The same B element is read once from SMEM, held in a register, and multiplied against 8 different A values — each producing a different partial sum. This amplifies the effective SMEM bandwidth by 8× without changing the HBM→SMEM tile loading pattern at all.

The general principle that every subsequent step inherits:

```
HBM tile  →  SMEM tile  →  Register tile  →  Tensor Core fragment
(global)    (block)         (thread)         (warp / wgmma)
```

Each level of tiling unlocks reuse at the next level of the hierarchy. CUTLASS systematizes this: every tile size is a template parameter. WGMMA / Tensor Core kernels add one more layer (the fragment tile fed to `wgmma.mma_async`). Same idea, more layers — every extra tile level absorbs another bandwidth bottleneck.

### Autotune result (2026-05-30)

Implemented `Matmul1DBlocktileAuto` (see `matmul_1d_blocktile.cu`) — first time `execute()` is called, it sweeps 7 legal `(BM, BK, TM)` candidates and caches the best for subsequent launches. The benchmark harness already takes a median over 100 iterations, so the one-time sweep cost gets absorbed.

**Kernel constraint exposed by the sweep**: the original 1D blocktile uses a 1-element-per-thread load (`innerRowA = tid/BK`, `innerColA = tid%BK`), which requires `NUM_THREADS == BM*BK == BK*BN == (BM*BN)/TM`. This forces **BM = BN and BM = BK·TM**, eliminating asymmetric tiles from the candidate grid. A truly general 1D blocktile would need a strided load like 2D blocktile uses; we deferred that change to keep this autotune step minimal.

**Result at N=4096**:

| Candidate | BM=BN | BK | TM | Threads | Outputs/thread | TFLOPS |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 4 | 8 | 128 | 8 | 15.47 |
| 1 | 32 | 8 | 4 | 256 | 4 | 15.14 |
| **2 (BEST)** | **64** | **4** | **16** | **256** | **16** | **19.26** |
| 3 | 64 | 16 | 4 | 1024 | 4 | 12.13 |
| 4 (default) | 64 | 8 | 8 | 512 | 8 | 17.63 |
| 5 | 128 | 8 | 16 | 1024 | 16 | 18.22 |
| 6 | 128 | 16 | 8 | (2048 > 1024) | — | skipped |

**Default vs Best comparison**:

| | Default (candidate 4) | Best (candidate 2) | Delta |
|---|---|---|---|
| `(BM=BN, BK, TM)` | `(64, 8, 8)` | `(64, 4, 16)` | — |
| Threads per block | 512 | 256 | ½× |
| Outputs per thread | 8 | 16 | 2× |
| SMEM per block | 4 KB | 2 KB | ½× |
| Register accumulators / thread | 8 | 16 | 2× |
| K-loop iterations (N=4096) | 512 | 1024 | 2× |
| Median time | 7.79 ms | 7.14 ms | −8.4% |
| **TFLOPS** | **17.64** | **19.26** | **+9.2%** |

The +9% jump came from a tile that's smaller in BK but bigger in TM — the opposite of "make tiles bigger". The winning config pushes the 1D blocktile core idea further: more register reuse per thread (16× instead of 8×). 256 threads/block is enough to hide latency, and the smaller SMEM footprint lets more blocks coexist per SM, keeping occupancy high.

**Bad config worth noting**: candidate 3 `(64, 16, 4)` is the *worst* at 12.13 TFLOPS even though it uses 1024 threads. With TM=4, each thread only computes 4 outputs → register reuse factor drops back near 1 → SMEM port pressure rises → throughput craters. This is direct empirical confirmation that **register reuse (TM), not thread count, is what drives 1D blocktile performance**.

**Algorithm ceiling visible**: even with the best 1D config, we only reach 19.3 TFLOPS — still well below 2D blocktile's 22.4 TFLOPS. 1D blocktile only reuses B in registers (not A), so even maximal TM cannot close the gap to 2D's `TM × TN = 64×` reuse. **Autotuning finds the best within an algorithm's ceiling; only a new algorithm raises the ceiling.**

See [`autotune.md`](autotune.md) for the autotune harness design and lessons learned, and [`blog-comparison-2026-05-30.md`](blog-comparison-2026-05-30.md) for the updated comparison table including the autotuned row.

---

## Step 5: 2D Blocktile — Register Reuse on Both A and B (Outer Product)

**File**: [`matmul_2d_blocktile.cu`](matmul_2d_blocktile.cu) | **Class**: `Matmul2DBlocktile`

### Core insight: 1D only reused B in registers. 2D reuses A too.

Recall 1D blocktile's inner loop:

```c
// 1D — only B is reused at register level
for (int dotIdx = 0; dotIdx < 8; dotIdx++) {
    float tmpB = Bs[dotIdx][threadCol];           // ← 1 B value in register
    for (int resIdx = 0; resIdx < 8; resIdx++) {
        threadResults[resIdx] +=
            As[threadRow*8 + resIdx][dotIdx]      // ← 8 fresh SMEM reads of A
            * tmpB;
    }
}
// 1 B SMEM read + 8 A SMEM reads = 9 SMEM reads → 8 madds
```

`tmpB` is held in a register and reused 8 times. Good. **But A is still read fresh from SMEM on every madd.** The SMEM-port pressure from A is still 1 read per madd — the same problem we solved for B is still alive on the A side.

2D blocktile fixes this by also pulling A's 8 values into registers, then doing an outer product:

```c
// 2D — both A and B held in registers, every madd is register-only
for (int dotIdx = 0; dotIdx < 8; dotIdx++) {
    // ① Load 8 A values into registers (one SMEM read each)
    for (int i = 0; i < TM; i++)
        regA[i] = As[threadRow*TM + i][dotIdx];   // 8 SMEM reads

    // ② Load 8 B values into registers
    for (int j = 0; j < TN; j++)
        regB[j] = Bs[dotIdx][threadCol*TN + j];   // 8 SMEM reads

    // ③ Outer product — 64 madds, all register-only
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            threadResults[i][j] += regA[i] * regB[j];
}
// 16 SMEM reads → 64 madds  (4× the arithmetic intensity of 1D)
```

### Outer product — why both sides get reused

For a fixed `dotIdx`, the 8 A values and 8 B values combine into an 8×8 outer product:

```
            regB[0..7]  ← 8 B values
           ┌──────────────────────┐
           │ b0  b1  b2 ... b7    │
           └──────────────────────┘

regA       threadResults[8][8]
[a0]       ┌──────────────────────┐
[a1]       │ a0·b0  a0·b1 ... a0·b7│   ← regA[0] reused across 8 columns
[a2]       │ a1·b0  a1·b1 ... a1·b7│   ← regA[1] reused across 8 columns
[..]   →   │   ...                │
[a7]       │ a7·b0  a7·b1 ... a7·b7│
           └──────────────────────┘
            ↑ regB[0] reused across 8 rows
```

Each `regA[i]` is multiplied against 8 different `regB[j]` values. Each `regB[j]` is multiplied against 8 different `regA[i]` values. **Both sides are reused 8 times → total reuse factor = 8 × 8 = 64.**

### Configuration

```
BM=128, BN=128, BK=8           ← block tile is 4× bigger than 1D
TM=8,   TN=8                    ← thread tile is now 8×8, not 8×1
256 threads/block               ← (128/8) × (128/8) = 16 × 16
threadResults[8][8]             ← 64 accumulators per thread (in registers)
regA[8], regB[8]                ← 16 input registers per thread
```

Each thread now owns a TM × TN = 8 × 8 = **64-element output square** of the block tile, instead of a 8 × 1 column slice.

### Memory hierarchy: where reuse happens

```
SMEM / L1 (block-level reuse — same as before)
   │
   │  Each iteration:
   │   - load 8 A values into regA[]  ← 8 SMEM reads
   │   - load 8 B values into regB[]  ← 8 SMEM reads
   ▼
Register File (thread-level reuse — BOTH A AND B)
   │
   │  Outer product:
   │   for i in 0..7:
   │     for j in 0..7:
   │       threadResults[i][j] += regA[i] * regB[j]
   │   → regA[i] reused 8 times (across j)
   │   → regB[j] reused 8 times (across i)
   │   → 64 madds from 16 SMEM reads
   ▼
threadResults[8][8] accumulators (in registers)
```

### Arithmetic intensity comparison

| Kernel | SMEM reads per inner iteration | madds per inner iteration | madd / SMEM read | What's reused in registers |
|---|---|---|---|---|
| SMEM | 2 (1A + 1B) | 1 | 0.50 | nothing |
| 1D blocktile | 9 (8A + 1B) | 8 | 0.89 | B only |
| **2D blocktile** | **16 (8A + 8B)** | **64** | **4.00** | **both A and B** |

SMEM-port pressure per madd: 2.0 (SMEM) → 1.12 (1D) → **0.25** (2D). That's an **8× reduction** vs SMEM kernel, with the same SMEM tile load pattern.

### Performance

| Kernel | TFLOPS | % vs cuBLAS FP32 | Improvement |
|---|---|---|---|
| 1D blocktile | 17.6 T | 33.7% | — |
| **2D blocktile** | **22.4 T** | **42.9%** | **+27%** |

The jump from 1D to 2D is smaller than from SMEM to 1D, because we already extracted most of the value at the first level of register reuse. But it confirms the principle: every additional dimension of register reuse buys real throughput.

### Why the load phase looks more complicated

256 threads must cover BM × BK = 128 × 8 = 1024 A elements (and 1024 B elements). That's 4 elements per thread. The kernel uses a strided load pattern:

```c
const int strideA = NUM_THREADS_2D / BK_2D;   // 256/8 = 32
for (int loadOffset = 0; loadOffset < BM_2D; loadOffset += strideA) {
    int row = innerRowA + loadOffset;
    As[row][innerColA] = A[row * N + innerColA];
}
```

Each thread loads 4 rows that are 32 apart, so within any single iteration of the `loadOffset` loop, the 32 threads of a warp all read the same row at consecutive columns — that's a coalesced 128B GMEM transaction. This separation between *load indexing* (`innerRowA`, `innerColA`) and *compute indexing* (`threadRow`, `threadCol`) is what lets us decouple "how many threads share the work of loading" from "how the thread tile is shaped in the output".

### What we learned

> **1D blocktile reuses B in registers (8 madds per SMEM read).
> 2D blocktile reuses BOTH A and B in registers via outer product (64 madds per SMEM read).
> Reuse factor = TM × TN. Every additional register-tile dimension multiplies the arithmetic intensity.**

The general pattern is now clear:

- **No register tile** (SMEM kernel): madd/read = 0.5
- **1D register tile** (TM=8): madd/read = TM / (TM+1) ≈ 1
- **2D register tile** (TM=8, TN=8): madd/read = TM·TN / (TM+TN) = 64/16 = 4
- **Tensor Core fragment** (16×16 or larger): madd/read = even higher

Going bigger on the thread tile (TM, TN) increases arithmetic intensity but also increases register pressure (TM×TN accumulators + TM + TN input registers). At some point you run out of registers per thread and occupancy collapses. The sweet spot for FP32 H100 is around TM=TN=8 (64 accumulators); WGMMA Tensor Core kernels push this to 64×N because the Tensor Core fragment itself replaces the inner 8×8 outer product with a single hardware instruction.

### Autotune result (2026-05-30)

Implemented `Matmul2DBlocktileAuto` (see `matmul_2d_blocktile.cu`). 15-candidate grid swept iteratively in two batches: 11 initial probes across `(BM, BN, BK, TM, TN)`, then 4 expansion probes informed by the first batch's winners. Verified across 3 independent runs — winner stable within ±0.2%, runner-up gap +3.6% (well above noise).

**Result at N=4096**:

| # | BM | BN | BK | TM | TN | Threads | SMEM | TFLOPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **10 (BEST)** | **128** | **128** | **16** | **16** | **8** | **128** | **16KB** | **34.03** |
| 2 | 128 | 128 | 8 | 16 | 8 | 128 | 8KB | 32.40 |
| 11 | 256 | 128 | 16 | 16 | 8 | 256 | 24KB | 31.54 |
| 8 | 256 | 128 | 8 | 16 | 8 | 256 | 12KB | 29.02 |
| 4 | 128 | 128 | 16 | 8 | 8 | 256 | 16KB | 28.55 |
| 13 | 128 | 128 | 16 | 8 | **16** | 128 | 16KB | 23.74 |
| 0 (default) | 128 | 128 | 8 | 8 | 8 | 256 | 8KB | 22.29 |
| 12 | 128 | 128 | **32** | 16 | 8 | 128 | 32KB | 22.47 |
| 7 | 128 | 128 | 8 | 4 | 4 | 1024 | 8KB | 21.50 |
| 14 | 256 | 256 | 16 | 16 | 8 | 512 | — | SKIPPED (register spill) |

(Full 19-row table in [`autotune.md`](autotune.md).)

**Default vs Best**:

| | Default (#0) | Best (#10) | Delta |
|---|---|---|---|
| `(BM, BN, BK, TM, TN)` | `(128, 128, 8, 8, 8)` | `(128, 128, 16, 16, 8)` | — |
| Threads per block | 256 | 128 | ½× |
| SMEM per block | 8 KB | 16 KB | 2× |
| Outputs per thread | 64 | 128 | 2× |
| Register accumulators / thread | 64 | 128 | 2× |
| K-loop iterations (N=4096) | 512 | 256 | ½× |
| Median time | 6.17 ms | 4.04 ms | −34% |
| **TFLOPS** | **22.29** | **34.03** | **+52.7%** |

The +52% jump pushes 2D blocktile from 42.7% to **65.2%** of cuBLAS FP32, closing most of the gap to Simon's autotuned A100 (68.7%) and beating Simon's autotuned warptile running on H100 (60.9%).

### Four lessons from the sweep

#### Lesson 1: BK = 16 is a sweet spot, not a monotonic ladder

```
BK=8   → 32.40 TFLOPS  (candidate 2)
BK=16  → 34.03 TFLOPS  (candidate 10) ← peak
BK=32  → 22.47 TFLOPS  (candidate 12) ← collapse
```

Going from BK=16 to BK=32 doubled SMEM per block (32KB vs 16KB) and pushed the SM toward fewer resident blocks. **Throughput dropped 33%.** The K-loop chunk size has to balance "more compute per sync" against "fewer blocks fit per SM" — BK=16 hits that balance on H100.

Earlier drafts of this doc cited a BK=24 datapoint at 27.7T. That number was **invalid** — the kernel's strided load requires `NUM_THREADS % BK == 0`, which BK=24 violates (`128 % 24 = 8`, `256 % 24 = 16`). The autotuner's validity check missed this, so candidates 15 and 17 were silently writing OOB into SMEM during the previous run. Gemini Code Assist caught this in PR review (commit 38f8709 added the missing divisibility checks); after the fix those candidates are correctly filtered out before launch. The valid data we have is BK ∈ {8, 16, 32}; the sweet-spot conclusion stands but we should not extrapolate to claims about BK=24.

#### Lesson 2: TM and TN are NOT mirror-symmetric

The most surprising finding:

```
TM=16, TN= 8 → 34.03 TFLOPS  (candidate 10)
TM= 8, TN=16 → 23.96 TFLOPS  (candidate 13) ← −30% from a mirror swap
```

Both configs have the same total register reuse factor (TM·TN = 128 madds per outer-product call), the same SMEM footprint, the same thread count. They differ only by which axis is "long".

Why the asymmetry? The compiler-emitted inner loop walks `for i { for j { regC[i][j] += regA[i] * regB[j] } }`. `regA[i]` is hoisted out of the inner j loop, so its lifetime spans TN iterations. With TM=16, TN=8, the j loop is short → `regA` register pressure is bounded → register allocation succeeds cleanly. With TM=8, TN=16, the j loop is longer → `regB[0..15]` must all be live simultaneously → register allocation gets tighter → likely spilling some accumulators to local memory.

This is one of those autotune findings that **no theory paper would tell you**. The kernel source code looks symmetric in TM and TN. The hardware behavior is not.

#### Lesson 3: Bigger block ≠ better when occupancy already saturates

```
BM=128, BN=128  →  34.03 TFLOPS  (candidate 10)
BM=256, BN=128  →  31.54 TFLOPS  (candidate 11)
BM=256, BN=256  →  REGISTER SPILL  (candidate 14)
```

Once the SM has enough concurrent blocks/warps to hide latency, making each block bigger just adds SMEM pressure without buying more parallelism. H100 has 132 SMs; at BM=BN=128, the 4096×4096 GEMM has 32×32 = 1024 blocks — plenty of work to spread across SMs without needing larger tiles.

#### Lesson 4: Default is one of the worst (again)

The hardcoded `(128, 128, 8, 8, 8)` siboehm-A100 default ranks **8th of 19** at 22.29T. Almost everything in the grid except the obvious bad configs (small TM/TN, mirror-swapped, oversized) beats the default. This re-confirms what 1D blocktile's autotune showed: H100 prefers smaller threads/block, larger thread tiles, and deeper BK than A100.

### Reproducibility check

Ran the full sweep 3 times to validate the winner isn't a noise artifact:

| Candidate | Run 1 | Run 2 | Run 3 | Spread |
|---|---:|---:|---:|---:|
| #10 (BEST) | 34.03 | 34.09 | 34.04 | ±0.06T (0.2%) |
| #2 (runner-up) | 32.65 | 32.91 | 32.84 | ±0.13T (0.4%) |
| #0 (default) | 22.31 | 22.42 | 22.45 | ±0.07T (0.3%) |

Winner-vs-runner-up gap (+3.6%) is 18× larger than the noise floor (~0.2%). Winner is real.

### Grid expansion methodology — iterative, not exhaustive

The first 11-candidate batch found `(128, 128, 16, 16, 8)` clustering at the top. The second batch added 4 candidates probing the boundaries of that cluster:

- candidate 11: push BM bigger → 31.8T (worse, occupancy hurt)
- candidate 12: push BK deeper → 22.2T (worse, SMEM bloat)
- candidate 13: mirror TM↔TN → 23.9T (worse, asymmetric finding)
- candidate 14: push both BM and BN bigger → register spill (kernel rejected)

All four expansions probed a boundary; none beat the winner. **The 4-axis sweet spot in the candidate grid coincides with a 4-axis wall**: BM=128 (occupancy wall), BK=16 (SMEM wall), TM=16 (register-reuse wall), TN=8 (asymmetric register-allocation wall). Pushing any direction degrades.

This is the kind of confidence empirical sweep gives you that paper-style theory cannot.

See [`autotune.md`](autotune.md) "Actual Results — 2D Blocktile" section for full sweep data, harness implementation notes, and updated status table.

---

## Step 6: Vectorized — `float4` Loads + Transposed A Tile

**File**: [`matmul_vectorized.cu`](matmul_vectorized.cu) | **Class**: `MatmulVectorized`

### Two changes bundled into one step

This step packages **two independent optimizations** that compound:

1. **`float4` vectorized GMEM loads and stores** — reduce instruction count for memory ops
2. **A tile transposed in SMEM** — eliminate SMEM bank conflicts and enable vectorized SMEM reads

Each one alone helps a little. Together they take us from 22.4T → 32.9T (**+47%, 63.0% vs cuBLAS FP32**).

### Change 1: `float4` for HBM ↔ register

Replace 4 scalar loads with 1 vector load:

```c
// Before — 4 instructions per thread to load 4 floats
float a0 = A[idx + 0];
float a1 = A[idx + 1];
float a2 = A[idx + 2];
float a3 = A[idx + 3];

// After — 1 instruction loads all 4 floats
float4 tmp = *reinterpret_cast<const float4*>(&A[idx]);
```

#### Why it helps

For a warp loading 32 floats:

| Method | Per-thread load | GMEM transactions | Instructions issued |
|---|---|---:|---:|
| Scalar `float` | 4 bytes | 1 × 128B per scalar load | 4 |
| `float4` | 16 bytes | 1 × 128B per scalar-equivalent (same total bytes) | **1** |

The total bytes moved are the same. The HBM-to-L2 path doesn't care which form you used. **What changes is the instruction count on the LSU** (Load/Store Unit). Each SM has only so many LSU pipes, and each pipe can issue one memory instruction per cycle. Cutting memory instructions 4× frees the LSU for compute or other memory ops in the same warp.

A second effect: each L1/L2 request carries metadata overhead (address, mask, cache tag lookup). Fewer requests = less cache controller pressure even when the same bytes are moved.

So vectorization is not about "moving more data" — it's about **issuing fewer instructions for the same data**.

### Change 2: A tile transposed in SMEM

```c
// 2D blocktile: A stored as [BM][BK]
__shared__ float As[BM_2D][BK_2D];        // [128][8]
regA[i] = As[threadRow*8 + i][dotIdx];    // stride-8 access

// Vectorized: A stored as [BK][BM] — transposed!
__shared__ float As[BK_VEC][BM_VEC];      // [8][128]
regA[i] = As[dotIdx][threadRow*8 + i];    // stride-1 access (contiguous)
```

The load phase writes A transposed into SMEM:

```c
// GMEM read: contiguous row of 4 floats
float4 tmp = *(...)(&A[innerRowA * N + innerColA * 4]);

// SMEM write: scatter the 4 values into 4 different rows of As
As[innerColA * 4 + 0][innerRowA] = tmp.x;
As[innerColA * 4 + 1][innerRowA] = tmp.y;
As[innerColA * 4 + 2][innerRowA] = tmp.z;
As[innerColA * 4 + 3][innerRowA] = tmp.w;
```

The compute phase then reads A's column from SMEM along a **contiguous** axis instead of strided.

### Why transpose matters here (and not earlier)

The most important conceptual point: **bank conflicts have been present since 2D blocktile, but only became the bottleneck after `float4` reduced GMEM instruction pressure.**

Walking through each kernel's SMEM access pattern from the warp's point of view:

| Step | A inner-loop access | Warp pattern | Bank conflict? | Why we didn't transpose |
|---|---|---|---|---|
| SMEM | `As[ty][k]` (ty=fixed in warp, k=fixed) | All 32 threads read the **same address** | None — pure broadcast | Not needed |
| 1D blocktile | `As[threadRow*8+resIdx][dotIdx]` (threadRow fixed in warp because BN=64 > 32) | All 32 threads read the **same address** | None — pure broadcast | Not needed |
| 2D blocktile | `As[threadRow*8+i][dotIdx]` (threadRow varies: warp spans 2 rows) | Warp splits into two 16-thread broadcast groups | 2-way conflict (mild) | Real but not the dominant bottleneck — GMEM instruction count and FP32 throughput were bigger limits |
| Vectorized (`float4`) | `As[dotIdx][threadRow*8+i]` | 16+16 broadcasts at non-conflicting banks | None | Now we transpose because the previous limit (GMEM instructions) was removed by `float4`, exposing SMEM as the next layer to fix |

#### Bank conflict mechanics in 2D blocktile

`threadRow = threadIdx.x / 16`, so a 32-thread warp has 16 threads with threadRow=0 and 16 with threadRow=1. For a fixed `dotIdx` and `i`:

- threadRow=0 group reads `As[0*8 + i][dotIdx]` — bank = `(0 + dotIdx) % 32`
- threadRow=1 group reads `As[1*8 + i][dotIdx]` — bank = `(64 + dotIdx) % 32` = `dotIdx % 32`

Both groups hit the **same SMEM bank**. SMEM bank conflicts serialize within a warp, so this is a 2-way conflict — the warp's SMEM read for A takes 2 cycles instead of 1. Not catastrophic; that's why 2D blocktile still hits 42.9% without fixing it.

#### Why `float4` exposes the conflict

Two things shift the bottleneck:

1. **GMEM load instruction count drops 4×** — the LSU is no longer saturated by tile loads, so the compute inner loop becomes a larger fraction of total runtime.
2. **The compute inner loop is dominated by SMEM reads** (16 reads → 64 madds). When the SMEM read takes 2 cycles instead of 1 due to bank conflict, that doubles the inner-loop SMEM time.

After step (1) frees the LSU, step (2)'s SMEM bank conflicts move from "background noise" to "30%+ of inner-loop latency." Transposing the A tile (a few extra lines of code in the load phase) fixes it cleanly.

### The general pattern: bottlenecks shift after each fix

This is the most important meta-insight from this step:

> **Optimization is layered. Fixing the current bottleneck reveals the next one. You don't fix everything at once because (a) you don't know which "issue" is actually a bottleneck until you've removed the larger ones, and (b) some optimizations only pay back after a prior layer is fixed.**

If we had transposed A in step 4 (1D blocktile) or step 5 (2D blocktile), the gain would have been small or zero — bank conflicts weren't on the critical path yet. Adding the transpose logic also slightly complicates the load phase (the scatter to 4 different SMEM rows), so doing it prematurely would add code complexity for negligible benefit.

This is the "premature optimization is the root of all evil" principle applied at the kernel level: **profile, find the true bottleneck, fix it, then re-profile to find the next one.**

### Configuration

```
BM=128, BN=128, BK=8           ← same block tile as 2D blocktile
TM=8,   TN=8                    ← same thread tile
256 threads/block               ← same thread count

NEW:
  As[BK][BM]                    ← transposed (was As[BM][BK])
  GMEM load: float4 (16-byte)   ← was scalar 4-byte
  GMEM store: float4            ← was scalar 4-byte
```

### Performance

| Kernel | TFLOPS | % vs cuBLAS FP32 | Improvement |
|---|---|---|---|
| 2D blocktile | 22.4 T | 42.9% | — |
| **Vectorized** | **32.9 T** | **63.0%** | **+47%** |

We've now beaten Simon's autotuned warptile running on H100 (31.8 TFLOPS) — without autotuning. This is also the first kernel to clear 60% of cuBLAS FP32.

### Autotune result (2026-05-31)

Implemented `MatmulVectorizedAuto` (see `matmul_vectorized.cu`) — first time `execute()` is called, sweeps 16 `(BM, BN, BK, TM, TN)` candidates and caches the best. Same structure as 2D blocktile (non-transposed `As[BM][BK]`, strided scalar loads) plus float4 (128-bit) C stores.

Performance comparison at N=4096 (idle pi1-h100-16, 3-run median):

| Variant | Config | N=4096 TFLOPS | vs cuBLAS FP32 |
|---|---|---|---|
| `vectorized` (hardcoded baseline) | `(128, 128, 8, 8, 8)` | 32.7 T | 62.7% |
| **`vectorized_auto` (winning config)** | **`(128, 128, 16, 16, 8)`** | **34.8 T** | **66.7%** |
| Delta | — | **+2.1T (+6.4%)** | **+4.0pp** |
| `2d_blocktile_auto` (prev winner) | `(128, 128, 16, 16, 8)` | 33.7 T | 64.6% |

Same winning config as 2D blocktile autotune (`(128, 128, 16, 16, 8)`). The float4 stores add +1.1T (+3.3%) on top of 2D auto's scalar stores — a modest but real win from reducing L1 cache-sector transactions during C writeback.

Notably candidates [1] (BK=8, TM=16, TN=8) and [13] (256×128, BK=16, TM=16, TN=8) also hit ~32.4T+, confirming the autotune space is well-sampled with a clear single peak. The minor gap between candidates [1] and [4] (BK=16 → TM=16/TN=8) mirrors what 2D autotune saw: deeper BK wins on H100 until SMEM occupancy saturates.

### What we learned

> **Vectorized GMEM (`float4`) cuts memory instruction count 4×, freeing the LSU. Transposing A in SMEM eliminates the bank conflicts that became visible once GMEM was no longer the dominant cost. The two changes only achieve their full +47% when bundled together.**

Two takeaways for future steps:

1. **Whenever you reduce one resource's pressure (here: LSU instructions), re-evaluate where the next bottleneck sits.** What looked like a small SMEM bank conflict in step 5 became a measurable 30%+ of inner-loop time in step 6 — same code, different relative cost.
2. **`float4` is universal** — every subsequent step (warptile, Tensor Core kernels) keeps `float4` GMEM loads. The transpose pattern also generalizes: any time SMEM access strides hit the same bank, transpose at the SMEM layout level rather than fix it at the indexing level.
