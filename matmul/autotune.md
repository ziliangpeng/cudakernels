# Autotuning — When and Why

**Date**: 2026-05-30
**Context**: companion to [`worklog.md`](worklog.md). This file captures the reasoning for *when* in the optimization journey autotuning starts to pay back, *what* dimensions open up at each step, and *how* a minimal autotune harness should look.

---

## The core question

> At which step does hardcoded tile sizes start to leave significant performance on the table, and where does autotuning become essential rather than optional?

Short answer: **autotuning becomes meaningful exactly when SMEM/register tiling begins**, because that's where the first real trade-off dimensions open up. Before SMEM tiling (naive, coalesced), there's almost nothing to tune.

---

## Why autotuning is the reverse of what we've been doing

Every kernel so far hardcodes its tile sizes:

```c
#define BM 128, BN 128, BK 8, TM 8, TN 8
```

These numbers came from siboehm's worklog on **A100**. We've been running them on H100. The [`blog-comparison-2026-05-30.md`](blog-comparison-2026-05-30.md) table shows the cost clearly:

| Step | Ours (H100, hardcoded) | Simon (A100, autotuned) | Gap |
|---|---|---|---|
| 2D blocktile | 42.9% | 68.7% | −25.8pp |
| Warptile | 54.2% | 93.7% | **−39.5pp** |

Our vectorized kernel actually **beats** Simon's autotuned warptile when both run on H100 (32.9 vs 31.8 TFLOPS) — proof that our algorithms are correct. The gap is purely parameter tuning.

---

## When each tunable dimension opens up

| Step | New tunable dimensions | Approx config count | Autotune ROI |
|---|---|---:|---|
| Naive | block dim (16×16 vs 32×32) | 2 | ~0% — algorithm dominates |
| Coalesced | block size (32 / 64 / … / 1024) | 6 | ~0% |
| **SMEM** | **+ tile size (16 / 32 / 64)** | **~10** | **~5%** — first real choice |
| **1D blocktile** | **+ BM, BN, BK, TM (decoupled)** | **~50** | **~10-15%** |
| **2D blocktile** | **+ TN** | **~200** | **+20-30%** |
| Vectorized | (same as 2D) | ~200 | +20-30% |
| **Warptile** | **+ WM, WN, WMITER, WNITER** | **~1000** | **+40-50%** — hardcoded almost always wrong |
| WGMMA / TC | + cluster shape, swizzle, pipeline depth | ~5000+ | +30%+ |

Two patterns:

1. **Each new tile dimension multiplies the config space.** By warptile, the manual search space is too large to reason through analytically.
2. **Each new dimension also unlocks new trade-offs.** BM↑ means bigger tile → more SMEM per block → fewer blocks per SM → lower occupancy. There's no single "right" answer; the optimum depends on the GPU's exact resource limits.

---

## Why autotune is genuinely educational (not just engineering polish)

The user's framing is correct: autotuning is not a separate phase you bolt on at the end for production. It's a learning tool.

What autotune teaches that hand-reasoning can't:

| Learning point | How autotune surfaces it |
|---|---|
| Which dimensions trade off against each other | Sweep one, hold others fixed — see the curve |
| Where hardware resource limits actually bind | Configs that don't compile / launch reveal hard limits (1024 threads/block, 228KB SMEM, 255 regs/thread, 64K regs/SM) |
| When occupancy matters vs doesn't | Sometimes low-occupancy + high-reuse wins; autotune finds these counterintuitive optima |
| Why H100 prefers different params than A100 | Direct comparison: same code, same N, different best config |
| Real bottleneck vs theoretical bottleneck | Profiler says "X is the limit"; autotune confirms or refutes by sweeping past it |

> **Without autotune, you only see what *should* work. With autotune, you see what *does* work — and the gap between those two is most of the lesson.**

---

## When NOT to autotune (yet)

For our specific worklog, the value-priority argument:

| Option | Expected gain |
|---|---|
| Autotune FP32 warptile | 28.3T → ~35T (+24%, ceiling = 52T) |
| WMMA → WGMMA (Tensor Core) | 27.5T → ~317T (+1050%, ceiling = 717T) |

WGMMA gives 40× more headroom. If the goal were purely speed, autotune FP32 would be deferred until after TC bring-up.

But the user's stated goal is **understanding**, not pure speed. And SMEM/blocktile autotune surfaces lessons that WGMMA bring-up doesn't. So the priority shifts:

- **Speed-only path**: defer autotune until everything algorithmic is done, then sweep once.
- **Learning path** (this project): autotune at each step that has tunable dimensions, *because the sweep itself teaches you something*.

We're on the learning path.

---

## Revised roadmap

```
Step 3   SMEM tiling                                          ✅ done
Step 3.5 Build minimal autotune harness                      ← new milestone
         Sweep SMEM_TILE = {16, 32}
         Confirm 1024-thread block limit blocks SMEM_TILE=64
Step 4   1D blocktile + autotune sweep BM/BN/BK/TM
Step 5   2D blocktile + autotune sweep BM/BN/BK/TM/TN
Step 6   Vectorized + autotune sweep
Step 7   Warptile + autotune sweep (autotune mandatory here)
Step 8+  Tensor Core path (WMMA → WGMMA → TMA → ...) + autotune at each
```

At each step:
1. Implement the new idea (hardcoded version)
2. Run autotune sweep
3. Record best config and best TFLOPS
4. Compare hardcoded vs autotuned — the delta is the "autotune ROI" data point
5. Document any surprising winners

---

## Lessons we expect autotune to teach at each step

### Step 3 — SMEM tiling (sweep `SMEM_TILE`)

Only 1 dimension. Three options:

| `SMEM_TILE` | Block threads | Result |
|---|---|---|
| 16 | 256 | smaller tile, more sync overhead, more blocks per SM, fewer registers per block |
| 32 | 1024 | current — at the 1024-thread block limit |
| 64 | 4096 | **won't launch** — exceeds CUDA 1024 thread/block hard cap |

Predicted lesson: **the 1024-thread block limit is exactly why we needed to invent the 1D blocktile concept** — to decouple "tile size" from "threads per block". The autotune sweep makes this constraint visible rather than abstract.

### Step 4 — 1D blocktile (sweep BM, BN, BK, TM)

Predicted lessons:
- Increasing BK (deeper K-direction load per iteration) reduces the loop trip count proportionally but increases SMEM usage per stage
- Increasing TM increases register pressure per thread (TM accumulators + 1 tmpB)
- Threads per block = (BM × BN) / TM — this must stay ≤ 1024
- Best config may have asymmetric BM ≠ BN

### Step 5 — 2D blocktile (sweep BM, BN, BK, TM, TN)

Predicted lessons:
- Symmetric tiles (TM = TN = 8) are usually but not always optimal
- TM × TN total register cost = TM × TN + TM + TN; pushes against 255-register limit
- H100 with 50MB L2 may prefer larger BM × BN than A100's 40MB L2

### Step 6 — Vectorized (same dimensions, expect new winners)

Predicted lesson: float4 changes the instruction-throughput vs SMEM trade-off, so the best tile config may shift from 2D's best.

### Step 7 — Warptile (sweep BM, BN, BK, TM, TN, WM, WN, WMITER, WNITER)

This is where autotune becomes mandatory. 9-dimensional space, hardcoded version is essentially guaranteed to be suboptimal. siboehm's 93.7% comes from autotuning here.

---

## Minimal harness design

### Approach: templated kernel + runtime dispatch + Python sweep

#### 1. Templatize the kernel

```cpp
template<int BM, int BN, int BK, int TM>
__global__ void matmul1DBlocktileKernelT(const float *A, const float *B, float *C, int N) {
    // body unchanged, but uses BM/BN/BK/TM as template params
}
```

#### 2. Runtime dispatch wrapper

```cpp
void launch1DBlocktile(const float *A, const float *B, float *C, int N,
                       int BM, int BN, int BK, int TM) {
    dim3 threads((BM * BN) / TM);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    if      (BM==64  && BN==64  && BK==8  && TM==8 ) matmul1DBlocktileKernelT<64,  64,  8,  8 ><<<blocks, threads>>>(A,B,C,N);
    else if (BM==64  && BN==128 && BK==8  && TM==8 ) matmul1DBlocktileKernelT<64,  128, 8,  8 ><<<blocks, threads>>>(A,B,C,N);
    else if (BM==128 && BN==128 && BK==8  && TM==8 ) matmul1DBlocktileKernelT<128, 128, 8,  8 ><<<blocks, threads>>>(A,B,C,N);
    // ... one branch per swept config
    else { fprintf(stderr, "Unsupported config\n"); exit(1); }
}
```

(Generate this dispatch table from a Python script — don't hand-write it. The dispatch list and the sweep list should come from the same source of truth.)

#### 3. CLI flag to select config at runtime

```bash
./matmul --kernel=1d_blocktile --BM=128 --BN=128 --BK=8 --TM=8
```

The benchmark harness already times one kernel — just pass the params through.

#### 4. Python sweep script

```python
import subprocess, itertools, re

configs = list(itertools.product(
    [64, 128],          # BM
    [64, 128],          # BN
    [8, 16],            # BK
    [4, 8, 16],         # TM
))

results = []
for BM, BN, BK, TM in configs:
    threads = (BM * BN) // TM
    if threads > 1024: continue                # CUDA hard limit
    if BM % TM != 0:   continue                # divisibility

    out = subprocess.check_output([
        "./matmul", "--kernel=1d_blocktile",
        f"--BM={BM}", f"--BN={BN}", f"--BK={BK}", f"--TM={TM}",
    ]).decode()
    tflops = float(re.search(r"TFLOPS=([\d.]+)", out).group(1))
    results.append((BM, BN, BK, TM, tflops))

results.sort(key=lambda r: -r[4])
for r in results[:10]:
    print(f"BM={r[0]:3d} BN={r[1]:3d} BK={r[2]:2d} TM={r[3]:2d}  →  {r[4]:.2f} TFLOPS")
```

#### 5. Validity-check first, then time

Skip configs that:
- Exceed 1024 threads per block
- Don't evenly divide N
- Allocate more than 228KB SMEM
- Use more than 255 registers per thread (visible from `nvcc --ptxas-options=-v`)

This avoids wasting time on configs that fail to launch.

---

## What to record

For each sweep, capture:

| Field | Why |
|---|---|
| `step` | Which kernel was swept |
| `N` (e.g. 4096) | Performance is N-dependent |
| All tile parameters | So configs are reproducible |
| Threads per block | Sanity check against 1024 limit |
| SMEM per block (bytes) | Sanity check against 228KB |
| Registers per thread (from PTXAS) | Sanity check against 255 |
| Resident blocks per SM (theoretical) | Occupancy estimate |
| TFLOPS | The metric |
| % vs cuBLAS FP32 | Normalized metric |

Output to a CSV per kernel step (e.g. `autotune_results_1d_blocktile.csv`), then point the worklog to it.

---

## Heuristics worth testing during autotuning

These should be hypotheses to check, not assumptions to encode:

1. **Power-of-2 tile sizes are usually best** (32, 64, 128, 256). H100 SM partition count is 4 and warp size is 32, so non-power-of-2 tile sizes often leave hardware idle.
2. **BK = 8 is a sweet spot for FP32** — small enough that 2 buffers fit in SMEM, large enough to amortize load cost.
3. **Larger BM × BN reduces block count → reduces tile-loading redundancy** but raises SMEM and register pressure.
4. **Asymmetric BM ≠ BN can help non-square matrices** but rarely helps square N×N.
5. **H100 prefers larger tiles than A100** because more SMs (132 vs 108), larger L2 (50MB vs 40MB), more SMEM per SM (228KB vs 164KB).

---

## What we'll learn from running this

The autotune sweep is the **first time** in this project where you'll see hardware constraints bite directly:

- "Why doesn't this config compile?" → 255 register limit
- "Why doesn't this config launch?" → 228KB SMEM limit, 1024 thread/block limit
- "Why is this config 30% slower despite using more SMEM?" → low occupancy
- "Why is the H100 best config different from A100?" → different cache/SM ratios

These insights don't come from reading more theory. They come from **running the sweep and looking at the results**. That's the educational value of autotuning at the SMEM/blocktile stage rather than deferring to the end.

---

## Related

- [`worklog.md`](worklog.md) — per-step optimization narrative
- [`blog-comparison-2026-05-30.md`](blog-comparison-2026-05-30.md) — gap analysis vs Simon (A100 autotuned) and Pranjal (H100 TC)
- [`../docs/ncu-profiling-2026-05-30.md`](../docs/ncu-profiling-2026-05-30.md) — Nsight profiling baseline
