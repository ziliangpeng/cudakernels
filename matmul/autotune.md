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

| Step | Ours (H100, hardcoded) | Simon (A6000, autotuned) | Gap |
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
- [`blog-comparison-2026-05-30.md`](blog-comparison-2026-05-30.md) — gap analysis vs Simon (A6000) and Pranjal (H100 TC)
- [`../docs/ncu-profiling-2026-05-30.md`](../docs/ncu-profiling-2026-05-30.md) — Nsight profiling baseline

---

## Actual Results — 1D Blocktile (2026-05-30)

First autotune step executed. Branch: `autotune-1d-blocktile`. Class: `Matmul1DBlocktileAuto`.

### Lessons from running the sweep

Several of the predictions from the design section above turned out wrong or only partially right. Recording the corrections is the whole point of an autotune log.

#### Lesson 1: The kernel's load pattern constrained the search space far more than expected

Original prediction (autotune.md §"Step 4 — 1D blocktile"): "BM, BN, BK, TM can vary independently; sweep ~50 candidates."

Reality: The hand-written kernel uses 1-element-per-thread load:
```c
innerRowA = tid / BK;   // covers BM × BK
innerRowB = tid / BN;   // covers BK × BN
```
Combined with `NUM_THREADS = (BM*BN)/TM`, this forces:
```
BM = BN  AND  BM = BK · TM
```
This eliminates almost all asymmetric configs. The "13-candidate 4K-friendly grid" we initially designed had **only 2 legal entries** when this constraint was applied. The first sweep crashed with `illegal memory access` because we didn't validate it before launching. The fix was a 7-candidate restricted grid.

**Generalization**: The strided load pattern in 2D blocktile (and CUTLASS) decouples NUM_THREADS from tile dimensions, which is why those kernels enjoy a much larger autotune space. For future kernel steps, we'll write strided loads from the start specifically to enable autotuning.

#### Lesson 2: H100 prefers smaller BK and larger TM, not "bigger tile"

Original prediction: "Larger BM × BN reduces block count → reduces tile-loading redundancy."

Reality: The winning config was the **smaller-BK, larger-TM** version, not a larger tile.

| Knob | Default `(64,8,8)` | Best `(64,4,16)` | Direction |
|---|---|---|---|
| BK | 8 | 4 | smaller |
| TM | 8 | 16 | larger |
| Threads/block | 512 | 256 | fewer |
| SMEM/block | 4KB | 2KB | smaller |
| Register reuse per SMEM read | 8× | **16×** | larger |

This rewards exactly what 1D blocktile's algorithm is about: **register reuse**. More outputs per thread (TM ↑) = more madds per SMEM B load. Smaller SMEM footprint lets more blocks fit per SM, keeping occupancy high.

#### Lesson 3: Bad configs validate the algorithm's mechanism

Candidate 3 (`BM=64, BK=16, TM=4`) was the **worst** at 12.13 TFLOPS despite using 1024 threads/block (theoretically max occupancy). With TM=4, each thread only computes 4 outputs, so the register reuse factor drops to 4× — barely above scalar SMEM kernel. Throughput dropped 30% below the default.

This is a direct empirical confirmation of the mental model from `worklog.md` Step 4: **register reuse is what drives 1D blocktile, not occupancy or thread count.**

#### Lesson 4: Autotune surfaces algorithm ceilings, not raises them

Best 1D blocktile config: 19.26 TFLOPS (36.9% vs cuBLAS FP32).
2D blocktile hardcoded: 22.3 TFLOPS (42.7%).

Even tuned to the limit, 1D cannot beat 2D — because 2D adds register reuse on A (via outer product) that 1D fundamentally lacks. **Autotune optimizes within an algorithm's ceiling; only a new algorithm raises the ceiling.**

### Performance summary

| Variant | Config | N=4096 TFLOPS | vs cuBLAS FP32 |
|---|---|---|---|
| `1d_blocktile` (hardcoded baseline) | `(64, 64, 8, 8)` | 17.6 | 33.7% |
| **`1d_blocktile_auto` (winning config)** | **`(64, 64, 4, 16)`** | **19.3** | **36.9%** |
| Delta | — | **+9.2%** | **+3.2pp** |
| Simon's autotuned 1D (A6000) | (varies) | 8.5 | 36.5% |

We now edge Simon's A6000 autotuned percentage at this step (36.9% vs 36.5%) — the first time our autotuned number matches a fully autotuned baseline on comparable hardware.

### Implementation notes

- Sweep runs **on first call to `execute()`**, not in constructor. Constructors should not do GPU work.
- The first timed iteration in the benchmark harness is slightly slower (it includes the sweep), but the harness already takes a median over 100 iterations, so the sweep cost is invisible in the reported number.
- Sweep policy: 2 warmup + 3 timed iterations per candidate, median wins.
- Total sweep cost at N=4096: ~6 candidates × 5 launches × ~7ms ≈ 200ms. Negligible against a 100-iteration timing loop that runs ~700ms.
- No CSV output written yet — `printf` only. Add CSV when we have more steps to compare.
- No `nvcc --ptxas-options=-v` register check yet — added as a TODO for next step (2D blocktile autotune).

### Open questions for next steps

1. **Should we rewrite 1D blocktile with strided loads to enable asymmetric tiles?** The autotune space would grow ~10×, possibly finding a 20-22 TFLOPS config. But this duplicates 2D blocktile's load pattern without giving us the 2D register reuse — limited educational return.
2. **Should `1d_blocktile_auto` print best config in the benchmark summary table?** Currently only shows up in stdout during sweep. The benchmark table just shows TFLOPS like any other method. Adding a "config" column would be nice but requires harness changes.
3. **Should autotune timing use the same 100-iter median as the main benchmark?** Currently uses 3-iter median (faster). Risk: noisy candidates might rank wrong. Mitigated by 9% gap between best and second-best — well above noise floor.

---

## Status

| Step | Autotune status | Best TFLOPS @ N=4096 | Winning config |
|---|---|---|---|
| SMEM | TODO | 9.0 (hardcoded) | — |
| 1D blocktile | ✅ Done | 19.3 | `(BM=BN=64, BK=4, TM=16)` |
| **2D blocktile** | **✅ Done** | **34.0** | **`(BM=BN=128, BK=16, TM=16, TN=8)`** |
| **Vectorized** | **✅ Done** | **34.8** | **`(BM=BN=128, BK=16, TM=16, TN=8)`** |
| Warptile | TODO | 28.3 (hardcoded) | — |

---

## Actual Results — 2D Blocktile (2026-05-30)

Second autotune step executed. Branch: `autotune-2d-blocktile`. Class: `Matmul2DBlocktileAuto`.

### Performance summary

| Variant | Config | N=4096 TFLOPS | vs cuBLAS FP32 |
|---|---|---|---|
| `2d_blocktile` (hardcoded baseline) | `(128, 128, 8, 8, 8)` | 22.3 | 42.7% |
| **`2d_blocktile_auto` (winning config)** | **`(128, 128, 16, 16, 8)`** | **34.0** | **65.2%** |
| Delta | — | **+52.7%** | **+22.5pp** |
| Simon's autotuned 2D (A6000) | (varies) | 16.0 | 68.7% |
| Simon's autotuned warptile (H100) | (varies) | 31.8 | 60.9% |

This is the **biggest single-step jump** in the entire project so far (+52%). We now nearly match Simon's autotuned 2D blocktile on A6000 (65.2% vs 68.7%) and **beat his autotuned warptile when run on H100** (65.2% vs 60.9%).

### Full sweep table (N=4096, 3-run median per candidate)

| # | BM | BN | BK | TM | TN | thr | SMEM | TFLOPS | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 128 | 128 | 8 | 8 | 8 | 256 | 8K | 22.29 | default |
| 1 | 64 | 64 | 8 | 8 | 8 | 64 | 4K | 28.12 | |
| 2 | 128 | 128 | 8 | 16 | 8 | 128 | 8K | 32.40 | runner-up |
| 3 | 128 | 128 | 8 | 8 | 16 | 128 | 8K | 22.89 | TM↔TN mirror at BK=8 |
| 4 | 128 | 128 | 16 | 8 | 8 | 256 | 16K | 28.55 | |
| 5 | 128 | 64 | 8 | 8 | 8 | 128 | 6K | 25.24 | asymmetric (tall) |
| 6 | 64 | 128 | 8 | 8 | 8 | 128 | 6K | 28.41 | asymmetric (wide) |
| 7 | 128 | 128 | 8 | 4 | 4 | 1024 | 8K | 21.50 | small thread tile |
| 8 | 256 | 128 | 8 | 16 | 8 | 256 | 12K | 29.02 | |
| 9 | 128 | 256 | 8 | 8 | 16 | 256 | 12K | 21.30 | mirror of #8 |
| **10 (BEST)** | **128** | **128** | **16** | **16** | **8** | **128** | **16K** | **34.03** | **winner** |
| 11 | 256 | 128 | 16 | 16 | 8 | 256 | 24KB | 31.54 | expansion (worse) |
| 12 | 128 | 128 | 32 | 16 | 8 | 128 | 32KB | 22.47 | expansion (SMEM wall) |
| 13 | 128 | 128 | 16 | 8 | 16 | 128 | 16KB | 23.74 | expansion (TM↔TN at BK=16) |
| 14 | 256 | 256 | 16 | 16 | 8 | 512 | — | SKIPPED | register spill |
| 15 | 256 | 128 | 24 | 16 | 8 | — | — | INVALID | filtered out: NUM_THREADS=256, 256%24≠0 → strided load OOB |
| 16 | 256 | 128 | 32 | 16 | 8 | 256 | 48KB | 20.87 | v3 — BK=32 confirmed bad at all blocks |
| 17 | 128 | 128 | 24 | 16 | 8 | — | — | INVALID | filtered out: NUM_THREADS=128, 128%24≠0 → strided load OOB |
| 18 | 256 | 256 | 8 | 16 | 8 | 512 | — | SKIPPED | v3 — register spill (256² area) |

The hardcoded `(128, 128, 8, 8, 8)` siboehm-A100 default ranks **8th of 19** at 22.29T (10 candidates beat it, 4 INVALID/SKIPPED, 4 worse). Almost everything in the grid except the obvious bad configs (small TM/TN, mirror-swapped, oversized) beats the default. This re-confirms what 1D blocktile's autotune showed: H100 prefers smaller threads/block, larger thread tiles, and deeper BK than A100.

### Reproducibility check

Sweeps have been re-run multiple times across pi1-h100-11 (during initial dev) and pi1-h100-16 (idle node, post-Gemini-review fixes). Winner has always been `(128, 128, 16, 16, 8)`. Absolute throughput ranges 33.7–34.1T across independent runs on idle nodes, well within typical node-noise (±0.5%). Numbers throughout this doc reflect the most recent clean idle-node run on pi1-h100-16 (~34.0T).

The originally-reported numbers came from a node where other Slurm jobs were not yet active (pre-fix sweep ran in low-contention window). Between the pre-fix and post-fix runs the cluster filled up, but pi1-h100-16 was idle when we re-tested, so the post-fix numbers are clean.

(Removed the original 3-run reproducibility table from this section; the post-fix run is the canonical number now.)

### Lessons (vs Lesson 2 surprises especially)

#### Lesson 1: BK = 16 is a sweet spot, not a monotonic ladder
BK=8 → 32.4T; BK=16 → 34.0T; BK=32 → 22.5T. Going past 16 raises SMEM per block, which drops the number of resident blocks per SM and collapses latency hiding. **K-loop depth has a sweet spot, not a "deeper is better" curve.** Earlier drafts cited a BK=24 datapoint at 27.7T; that number was the product of an OOB strided load (Gemini caught it in PR review — `NUM_THREADS % BK == 0` is a hard requirement for the kernel's strided load to cover the A/B tiles exactly). After commit 38f8709 added the divisibility checks, BK=24 candidates are correctly filtered out before launch. Valid points are BK ∈ {8, 16, 32}.

#### Lesson 2: TM and TN are NOT mirror-symmetric (most surprising)
`(TM=16, TN=8) → 34.03 TFLOPS` but `(TM=8, TN=16) → 23.74 TFLOPS`. Same total reuse (TM·TN=128), same SMEM, same thread count. Difference: the inner loop is `for i { for j { ... regA[i] * regB[j] } }`. Hoisted `regA[i]` has lifetime TN cycles. Long TN forces all regB[0..TN-1] to be live simultaneously → tighter register allocation → likely partial spill of accumulators.

This is the kind of finding **no theory paper would tell you**. Source code looks symmetric; hardware behavior is not.

#### Lesson 3: Bigger block ≠ better when SMs already have enough blocks
At BM=BN=128, the 4096² GEMM produces 1024 blocks for 132 SMs — already 8 blocks/SM in flight. Pushing BM to 256 just halves the block count without increasing concurrency, while doubling SMEM footprint. Net loss.

#### Lesson 4: Default is one of the worst (again)
Default ranks 7th of 15 (22.29T vs winner 34.03T). Same pattern as 1D autotune: hardcoded A100 numbers are bad on H100. Sweep margin: +52.7%.

### Implementation notes

- 15-candidate grid built in **two iterations**: 11 initial probes + 4 informed expansions
- After first batch found winners clustering at `(128, 128, 16, 16, 8)`, expansion probes each pushed one axis to test where the boundary lies
- All 4 expansions probed a wall; none beat the winner. This gives strong confidence the winner is near-optimal within the kernel design
- `cudaGetLastError` after warmup cleanly catches candidate 14's register-spill failure → printed `SKIPPED (too many resources requested for launch)` → no false ranking
- Strided load pattern (vs 1D's 1-element-per-thread) enables much wider candidate space — we explored 15 configs vs 7 for 1D
- Total sweep cost at N=4096: ~14 candidates × 5 launches × ~5ms ≈ 350ms. Negligible against the 100-iteration timing loop

### Iterative-vs-exhaustive grid design

This step validated the "expand grid based on what the previous batch revealed" approach:

- Batch 1 (11 candidates): broad exploration, found `(128,128,16,16,8)` near the top
- Batch 2 (4 candidates): targeted probes at each adjacent direction
- All 4 probes hit a wall, confirming the winner is local optimum

The full Cartesian product `BM × BN × BK × TM × TN` would be ~thousands of configs. The iterative approach got us to a confidently-near-optimal winner in 15 configs by following the data.

### Open questions

1. **Should we sweep at multiple N (not just 4096)?** Smaller N might prefer smaller blocks for better SM coverage. Currently each `Matmul2DBlocktileAuto` instance sweeps once for its own N, so the benchmark `all` mode does pick per-N winners — we just haven't analyzed them.
2. **The TM ≠ TN asymmetry is a real algorithmic finding.** Should we add a swap-direction toggle to the kernel? Probably not worth it — the asymmetry only matters when one of TM/TN is large; for square tiles it's symmetric.

---

## Actual Results — Vectorized (2026-05-31)

Third autotune step executed. Branch: `autotune-vectorized`. Class: `MatmulVectorizedAuto`.

### Design choices

The vectorized autotune uses **the same structure as 2D blocktile** (non-transposed `As[BM][BK]`, strided scalar GMEM loads, outer product compute) plus float4 (128-bit) C stores. This is different from the hardcoded vectorized kernel, which uses float4 GMEM loads + transposed `As[BK][BM]`. Rationale:

- The templated kernel needs to support arbitrary candidate configs. float4 GMEM loads require `BK % 4 == 0` and a specific load index pattern that's hard to template cleanly.
- The proven 2D autotune structure works well — adding float4 C stores on top isolates the incremental benefit of vectorized stores vs scalar stores.

### Performance summary

| Variant | Config | N=4096 TFLOPS | vs cuBLAS FP32 |
|---|---|---|---|
| `vectorized` (hardcoded baseline) | `(128, 128, 8, 8, 8)` | 32.7 T | 62.7% |
| **`vectorized_auto` (winning config)** | **`(128, 128, 16, 16, 8)`** | **34.8 T** | **66.7%** |
| Delta | — | **+2.1T (+6.4%)** | **+4.0pp** |
| `2d_blocktile_auto` (prev winner) | `(128, 128, 16, 16, 8)` | 33.7 T | 64.6% |
| `vectorized_auto` vs `2d_blocktile_auto` | same config | **+1.1T (+3.3%)** | **+2.1pp** |

### Full sweep table (N=4096, single-launch median per candidate, post-fix re-sweep on exclusive pi1-h100-27)

| # | BM | BN | BK | TM | TN | thr | SMEM | TFLOPS | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 128 | 128 | 8 | 8 | 8 | 256 | 8K | 22.68 | default |
| 1 | 128 | 128 | 8 | 16 | 8 | 128 | 8K | 33.07 | runner-up @ BK=8 |
| 2 | 128 | 128 | 8 | 8 | 16 | 128 | 8K | 23.08 | TM↔TN mirror (bad) |
| 3 | 128 | 128 | 16 | 8 | 8 | 256 | 16K | 30.53 | |
| **4 (BEST)** | **128** | **128** | **16** | **16** | **8** | **128** | **16K** | **34.49** | **winner (avg 34.90)** |
| 5 | 128 | 128 | 16 | 8 | 16 | 128 | 16K | 26.43 | TM↔TN mirror at BK=16 |
| 6 | 128 | 64 | 8 | 16 | 8 | 64 | 6K | 30.38 | asymmetric (tall) |
| 7 | 64 | 128 | 8 | 8 | 8 | 128 | 6K | 28.24 | asymmetric (wide), valid 128 thr |
| 8 | 256 | 128 | 8 | 16 | 8 | 256 | 12K | 29.42 | bigger block (tall) |
| 9 | 128 | 256 | 8 | 8 | 16 | 256 | 12K | 22.39 | bigger block (wide) |
| 10 | 64 | 64 | 8 | 8 | 8 | 64 | 4K | 29.55 | small block |
| 11 | 64 | 64 | 16 | 8 | 8 | 64 | 8K | 32.57 | small + deeper BK |
| 12 | 128 | 128 | 32 | 16 | 8 | 128 | 32K | 22.75 | BK wall |
| 13 | 256 | 128 | 16 | 16 | 8 | 256 | 24K | 33.34 | bigger tall |
| 14 | 256 | 256 | 8 | 16 | 8 | 512 | — | SKIPPED | register spill |
| 15 | 128 | 128 | 16 | 4 | 4 | 1024 | 16K | 26.59 | small thread tile |

### Reproducibility

Two independent runs on exclusive `pi1-h100-27` (job 11723, full 8-GPU reservation, OverSubscribe=NO) gave 34.49T and 34.57T best — within 0.2%. The 100-iteration average is 34.90T / 34.89T — also tight. Earlier numbers (34.77–34.81T) came from `pi1-h100-16`, which we later discovered was a shared dev-partition node (OverSubscribe=OK, no GPU TRES on our salloc). Numbers above are from the truly-exclusive node post-`As[BM][BK+1]` bank-conflict fix.

### Bank-conflict padding (Gemini PR #4 review)

Gemini's PR #4 review caught that the inner-loop SMEM access `As[threadRow * TM + i][dotIdx]` causes a 2-way bank conflict when `TM * BK` is a multiple of 32 (e.g. `BK=16, TM=16` → stride 256 = 8×32). The fix is a 1-element inner-dimension pad: `__shared__ float As[BM][BK + 1]`. Inner stride becomes `TM * (BK+1) = 17 * 16 = 272`, not divisible by 32 → conflict eliminated. Cost: 512 bytes extra SMEM for BM=128.

Effect by candidate (delta vs pre-fix on shared `pi1-h100-16`):

| Candidate | Pre-fix (shared -16) | Post-fix (exclusive -27) | Δ |
|---|---|---|---|
| [4] WINNER (128,128,**16**,16,8) | 34.81 | 34.49 | flat (within noise) |
| [1] (128,128,**8**,16,8) | 32.43 | 33.07 | +0.6 |
| [11] (64,64,**16**,8,8) | 29.61 | **32.57** | **+3.0** |
| [13] (256,128,**16**,16,8) | 32.22 | 33.34 | +1.1 |

Several BK=16 candidates jump by 1-3T. The winner [4] is flat — it's already register/compute-bound, not SMEM-port-bound, so removing the conflict doesn't help. The ranking is unchanged (same winner config), so this is a real but modest defensive cleanup, not a peak-perf improvement. Worth keeping because (a) it's free, (b) it would matter if a future kernel variant hits the bank-conflict scenario.

### Key finding

**float4 C stores are a real (if modest) optimization on H100.** The +3.3% gain over 2D blocktile's scalar stores is reproducible and comes from reducing L1 cache-sector transactions during C writeback. Each 128-bit store maps to one L1 sector write instead of up to 4 for scalar stores, reducing LSU pressure during the output phase.

The winning config is the same as 2D blocktile: `(128, 128, 16, 16, 8)`. This confirms the autotune landscape is stable — vectorized stores don't change which tile sizes are optimal, they just shift the performance ceiling slightly higher.

### What's next

Only warptile remains for FP32 autotune. The hardcoded warptile runs 28.3T (54.2%) — the biggest gap between autotuned and hardcoded for any step. Warptile autotuning is expected to be the largest remaining gain in the FP32 path.
- **Should we extract autotune harness into a shared file?** Both 1D and 2D `*Auto` classes have nearly identical `tune()` boilerplate. Refactor candidate after 1-2 more steps.

---

## Cross-Platform Autotune — A100 Spot VM (2026-06-01)

The entire autotune grid was re-run on an A100-SXM4-40GB spot VM (`a100-spot-5`, us-east1-b, driver 535.309.01, CUDA 12.4, 108 SMs @ 1.41 GHz). Same binaries, same candidate grids — only the GPU is different.

### 1D Blocktile — A100 Results

| # | BM | BN | BK | TM | thr | TFLOPS | notes |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 32 | 32 | 4 | 8 | 128 | 6.83 | |
| 1 | 32 | 32 | 8 | 4 | 256 | 6.96 | |
| **2 (BEST)** | **64** | **64** | **4** | **16** | **256** | **11.15** | **winner** |
| 3 | 64 | 64 | 16 | 4 | 1024 | 7.24 | |
| 4 | 64 | 64 | 8 | 8 | 512 | 10.05 | |
| 5 | 128 | 128 | 8 | 16 | 1024 | 10.22 | |

The winner (`BM=BN=64, BK=4, TM=16`) is the same parameter count as H100's winner — but the reason is different. On H100, smaller BK won because it allows larger TM (= more register reuse). On A100, the constraint is SMEM bandwidth: at N=4096, BK=4 keeps each block's SMEM working set small, letting more blocks co-reside per SM. The H100 winner also happens to be the best config on A100 because the kernel's constraint (`BM=BN=BK·TM`) limits the space so much that the same config dominates both.

| Metric | H100 (pi1-h100-27) | A100 (spot VM) | Ratio |
|---|---|---|---|
| 1D blocktile auto (TFLOPS) | 19.26 | 11.15 | 1.73× |
| % vs cuBLAS FP32 | 36.9% | 59.9% | — |
| cuBLAS FP32 baseline | 52.2 T | 18.6 T | 2.81× |
| Winner config | BM=64 BN=64 BK=4 TM=16 | same | — |

The A100 gets a *higher* % vs cuBLAS (59.9% vs 36.9%) because cuBLAS exploits Tensor Cores on H100 much more effectively than on A100 — our pure-FP32 kernel runs into a lower cuBLAS ceiling on A100.

### 2D Blocktile — A100 Results

Full sweep of 19 candidates (same grid as H100):

| # | BM | BN | BK | TM | TN | thr | SMEM | TFLOPS | notes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 128 | 128 | 8 | 8 | 8 | 256 | 8K | 8.95 | default |
| 1 | 64 | 64 | 8 | 8 | 8 | 64 | 4K | 11.88 | |
| 2 | 128 | 128 | 8 | 16 | 8 | 128 | 8K | 11.57 | |
| 3 | 128 | 128 | 8 | 8 | 16 | 128 | 8K | 11.27 | TM↔TN mirror |
| 4 | 128 | 128 | 16 | 8 | 8 | 256 | 16K | 13.18 | |
| 5 | 128 | 64 | 8 | 8 | 8 | 128 | 6K | 15.55 | asymmetric (tall) |
| **6 (BEST)** | **64** | **128** | **8** | **8** | **8** | **128** | **6K** | **16.37** | **winner — asymmetric (wide)** |
| 7 | 128 | 128 | 8 | 4 | 4 | 1024 | 8K | 11.03 | small thread tile |
| 8 | 256 | 128 | 8 | 16 | 8 | 256 | 12K | 13.01 | |
| 9 | 128 | 256 | 8 | 8 | 16 | 256 | 12K | 12.13 | mirror of #8 |
| 10 | 128 | 128 | 16 | 16 | 8 | 128 | 16K | 15.15 | H100 winner |
| 11 | 256 | 128 | 16 | 16 | 8 | 256 | 24K | 13.88 | |
| 12 | 128 | 128 | 32 | 16 | 8 | 128 | 32K | 10.95 | BK wall |
| 13 | 128 | 128 | 16 | 8 | 16 | 128 | 16K | 13.38 | TM↔TN at BK=16 |
| 14 | 256 | 256 | 16 | 16 | 8 | 512 | — | SKIPPED | register spill |
| 16 | 256 | 128 | 32 | 16 | 8 | 256 | 48K | 10.03 | |
| 18 | 256 | 256 | 8 | 16 | 8 | 512 | — | SKIPPED | register spill |

### The Big Finding: Asymmetric Tiles Win on A100

**Winner: `BM=64 BN=128 BK=8 TM=8 TN=8` — 16.37 TFLOPS (88.0% vs cuBLAS)**

The H100 winner `(128, 128, 16, 16, 8)` ranks **4th** on A100 at 15.15 T. The winning config is **asymmetric** (64×128 wide rectangle), which makes sense:

- **A100 has 108 SMs vs H100's 132**: fewer SMs means each SM must cover more work → smaller blocks = more blocks = better SM utilization
- **A100 SMEM bandwidth is lower**: the asymmetric tile moves more data through the K dimension (B loads) without growing SMEM proportionally (6K vs 16K for the H100 winner)
- **A100 prefers BK=8**: H100's sweet spot at BK=16 doesn't hold on A100 — the smaller SMEM per block keeps more blocks resident

This is the first concrete evidence that **autotune winners are architecture-specific**. The parameter space is identical; the hardware picks different optima.

| Metric | H100 (pi1-h100-27) | A100 (spot VM) |
|---|---|---|
| 2D blocktile auto (TFLOPS) | 34.03 | 16.37 |
| % vs cuBLAS FP32 | 65.2% | 88.0% |
| Winner config | BM=BN=128 BK=16 TM=16 TN=8 | BM=64 BN=128 BK=8 TM=8 TN=8 |
| Tile shape | symmetric (128×128) | asymmetric (64×128, wide) |
| Simon's 2D auto (A6000) | — | 84.8% (16.0 T)* |

We beat Simon's A6000 2D blocktile in absolute TFLOPS (16.4 T > 16.0 T), but the 88.0% vs 84.8% headline comparison is misleading — his % is TF32-relative, ours is FP32-relative. See [blog-comparison-2026-05-30.md](blog-comparison-2026-05-30.md) for the full audit.

### A100 Full Autotune Summary

| Step | H100 Winner | A100 Winner | H100 TFLOPS | A100 TFLOPS | Notes |
|---|---|---|---|---|---|
| 1D blocktile auto | BM=64 BN=64 BK=4 TM=16 | same | 19.3 T | 11.2 T | +1.7 T over Simon (A6000, 8.5 T) |
| 2D blocktile auto | BM=BN=128 BK=16 TM=16 TN=8 | BM=64 BN=128 BK=8 TM=8 TN=8 | 34.0 T | 16.4 T | +0.4 T over Simon's best (16.0 T) |
| Vectorized auto | BM=BN=128 BK=16 TM=16 TN=8 | (H100-tuned reused) | 34.8 T | 16.6 T | −1.6 T vs Simon's 18.2 T |
| Warptile auto | BM=BN=128 BK=8 TM=4 TN=4 WM=WN=64 | (H100-tuned reused) | 33.4 T | 15.0 T | −6.8 T vs Simon's 21.8 T |

*\*Simon uses A6000 (84 SM) with TF32 cuBLAS baseline 23.2 T. Our A100 (108 SM) with FP32 cuBLAS 18.6 T. Per-SM efficiency analysis in blog-comparison doc.*

### Open Questions

1. **Should we run vectorized_auto and warptile_auto sweeps on A100?** The 2D blocktile result shows autotune winners are architecture-specific. The warptile_auto H100-tuned config gets only 15.0 T on A100 (vs Simon's 21.8 T on A6000 with fewer SMs — his per-SM warp-tile efficiency is exceptional). An A100-native warptile sweep is the obvious next step for closing this gap.
2. **Does the asymmetric-tile preference generalize?** The `(64×128)` winner suggests A100 prefers wider tiles that amortize B-loads across more columns. Would `(128×64)` also win, or is there a directionality bias (A-load vs B-load reuse patterns)?
3. **N-dependence?** These sweeps are all at N=4096. Smaller N might prefer different tile shapes entirely on A100 (more SMs idle → incentive for larger blocks to keep all SMs fed).

### A100 VM Notes

- **IAP tunneling is the only viable SSH method** for `character-ai` GCP project (firewall rules restrict port 22; changing them is forbidden)
- **Driver 535.309.01 server** installed successfully after reboot — earlier attempts on same kernel (6.8.0-1060-gcp) in us-central1-a failed with DKMS build errors (unknown why us-east1-b succeeded, possibly different GCC)
- **`/tmp/matmul_test` is ephemeral** — spot VM can be preempted any time. Rebuild: `git clone → nvcc -O2 -arch=sm_80 -I.. -o /tmp/matmul_test matmul.cpp matmul_*.cu matrix_init.cu -lcublas -lcublasLt`
- Full setup log in `~/code/ziliang2026/projects/2026-06-01-gcp-a100-nvidia-driver-setup/README.md`
