#include "matmul_2d_blocktile.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

// ============================================================================
// Templated 2D Block Tiling kernel
// ============================================================================
//
// Each thread computes a TM x TN tile of output elements via outer product:
//     regC[i][j] += regA[i] * regB[j]    for i in 0..TM-1, j in 0..TN-1
//
// Strided load pattern decouples NUM_THREADS from tile dimensions, so this
// kernel works for any (BM, BN, BK, TM, TN) where:
//   - threads_per_block = (BM/TM) * (BN/TN) <= 1024
//   - threads_per_block divides BM*BK (A tile element count)
//   - threads_per_block divides BK*BN (B tile element count)
//
// All boundary checks preserved for arbitrary N.

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmul2DBlocktileKernelT(const float *A, const float *B, float *C, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += blockRow * BM * N;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    float threadResults[TM][TN] = {{0.0f}};
    float regA[TM];
    float regB[TN];

    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr int strideA = NUM_THREADS / BK;
    constexpr int strideB = NUM_THREADS / BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK) {
        // Load A tile (strided). For each candidate we ensure strideA divides BM,
        // so loadOffset never exceeds BM and no inner row-bounds check is needed.
        #pragma unroll
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            int row = innerRowA + loadOffset;
            if (blockRow * BM + row < N && tileIdx + innerColA < N) {
                As[row][innerColA] = A[row * N + innerColA];
            } else {
                As[row][innerColA] = 0.0f;
            }
        }

        // Load B tile (strided). strideB divides BK by candidate validity.
        #pragma unroll
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            int row = innerRowB + loadOffset;
            if (tileIdx + row < N && blockCol * BN + innerColB < N) {
                Bs[row][innerColB] = B[row * N + innerColB];
            } else {
                Bs[row][innerColB] = 0.0f;
            }
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // Outer product compute
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                regA[i] = As[threadRow * TM + i][dotIdx];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regB[j] = Bs[dotIdx][threadCol * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int globalRow = blockRow * BM + threadRow * TM + i;
            int globalCol = blockCol * BN + threadCol * TN + j;
            if (globalRow < N && globalCol < N) {
                C[(threadRow * TM + i) * N + threadCol * TN + j] = threadResults[i][j];
            }
        }
    }
}

// ============================================================================
// Original (hardcoded) Matmul2DBlocktile — kept as baseline for comparison.
// Uses BM=BN=128, BK=8, TM=TN=8 (matches siboehm Kernel 5 defaults).
// ============================================================================

#define BM_2D 128
#define BN_2D 128
#define BK_2D 8
#define TM_2D 8
#define TN_2D 8
#define NUM_THREADS_2D ((BM_2D / TM_2D) * (BN_2D / TN_2D))

Matmul2DBlocktile::Matmul2DBlocktile(int N, int blockDim) : N(N), blockDim(blockDim) {}

void Matmul2DBlocktile::execute(const float *d_A, const float *d_B, float *d_C) {
    dim3 threads(NUM_THREADS_2D);
    dim3 blocks((N + BN_2D - 1) / BN_2D, (N + BM_2D - 1) / BM_2D);
    matmul2DBlocktileKernelT<BM_2D, BN_2D, BK_2D, TM_2D, TN_2D><<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());
}

Matmul2DBlocktile::~Matmul2DBlocktile() {}

// ============================================================================
// Autotuning version — Matmul2DBlocktileAuto
// ============================================================================
//
// 19-candidate sweep across (BM, BN, BK, TM, TN). Each candidate satisfies:
//   - NUM_THREADS = (BM/TM) * (BN/TN) <= 1024 (CUDA hard limit)
//   - NUM_THREADS divides BM*BK (A tile load element count)
//   - NUM_THREADS divides BK*BN (B tile load element count)
//   - NUM_THREADS divides BK and BN cleanly (so strideA = NUM_THREADS/BK divides BM)
//   - (BM*BK + BK*BN) * 4 bytes <= 48KB SMEM (default static SMEM limit, no opt-in)
//   - TM*TN + TM + TN <= ~96 (rough register budget)
//
// The sweep itself catches any post-warmup CUDA errors and skips those configs.

struct Candidate2D {
    int BM, BN, BK, TM, TN;
};

static const Candidate2D CANDIDATES_2D[] = {
    // {BM, BN, BK, TM, TN}  SMEM = (BM*BK + BK*BN)*4 bytes (single-buffered)
    {128, 128,  8,  8,  8},   // [ 0] CURRENT DEFAULT — 256 thr, 64 acc, 8KB SMEM
    { 64,  64,  8,  8,  8},   // [ 1] smaller block — 64 thr, 4KB SMEM, more blocks/SM
    {128, 128,  8, 16,  8},   // [ 2] bigger TM — 128 thr, 128 acc, 8KB SMEM
    {128, 128,  8,  8, 16},   // [ 3] bigger TN — 128 thr, 128 acc, 8KB SMEM
    {128, 128, 16,  8,  8},   // [ 4] deeper BK — 16KB SMEM, fewer K-iters
    {128,  64,  8,  8,  8},   // [ 5] asymmetric (tall) — 128 thr, 6KB SMEM
    { 64, 128,  8,  8,  8},   // [ 6] asymmetric (wide) — 128 thr, 6KB SMEM
    {128, 128,  8,  4,  4},   // [ 7] small thread tile — 1024 thr, 16 acc, 8KB SMEM
    {256, 128,  8, 16,  8},   // [ 8] bigger block (tall) — 128 thr, 12KB SMEM
    {128, 256,  8,  8, 16},   // [ 9] bigger block (wide) — 128 thr, 12KB SMEM
    {128, 128, 16, 16,  8},   // [10] deepest BK + bigger TM — 128 thr, 16KB SMEM
    // --- Grid expansion v2 (informed by first sweep) ---
    // First sweep showed winners cluster around (large TM, deeper BK).
    // These probes test how far that trend extrapolates before hitting other
    // resource walls (SMEM port pressure, register count, K-loop overhead).
    {256, 128, 16, 16,  8},   // [11] push #10 winner to bigger block (tall) — 24KB SMEM
    {128, 128, 32, 16,  8},   // [12] does BK=32 keep winning, or oversaturate? — 32KB SMEM
    {128, 128, 16,  8, 16},   // [13] mirror of #10 — TM↔TN swap, same total reuse — 16KB SMEM
    {256, 256, 16, 16,  8},   // [14] big square block + deepest BK + best TM/TN — 32KB SMEM
    // --- Grid expansion v3 (after SMEM math fix; we had been overestimating SMEM 2x,
    //     so larger blocks are actually well within the 228KB limit) ---
    {256, 128, 24, 16,  8},   // [15] deeper BK + larger block (tall) — 36KB SMEM
    {256, 128, 32, 16,  8},   // [16] BK=32 on larger block — 48KB SMEM
    {128, 128, 24, 16,  8},   // [17] BK between 16 and 32 — 24KB SMEM
    {256, 256,  8, 16,  8},   // [18] big square block, shallow BK — 16KB SMEM
};
static const int NUM_CANDIDATES_2D = sizeof(CANDIDATES_2D) / sizeof(CANDIDATES_2D[0]);

// Dispatch table — explicit template instantiations.
// One branch per candidate. Order MUST match CANDIDATES_2D[].
void Matmul2DBlocktileAuto::launch(const float *d_A, const float *d_B, float *d_C,
                                   int BM, int BN, int BK, int TM, int TN) {
    int threads_per_block = (BM / TM) * (BN / TN);
    dim3 threads(threads_per_block);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    #define DISPATCH(_BM, _BN, _BK, _TM, _TN) \
        if (BM == _BM && BN == _BN && BK == _BK && TM == _TM && TN == _TN) { \
            matmul2DBlocktileKernelT<_BM, _BN, _BK, _TM, _TN><<<blocks, threads>>>(d_A, d_B, d_C, N); \
            return; \
        }

    DISPATCH(128, 128,  8,  8,  8)
    DISPATCH( 64,  64,  8,  8,  8)
    DISPATCH(128, 128,  8, 16,  8)
    DISPATCH(128, 128,  8,  8, 16)
    DISPATCH(128, 128, 16,  8,  8)
    DISPATCH(128,  64,  8,  8,  8)
    DISPATCH( 64, 128,  8,  8,  8)
    DISPATCH(128, 128,  8,  4,  4)
    DISPATCH(256, 128,  8, 16,  8)
    DISPATCH(128, 256,  8,  8, 16)
    DISPATCH(128, 128, 16, 16,  8)
    DISPATCH(256, 128, 16, 16,  8)
    DISPATCH(128, 128, 32, 16,  8)
    DISPATCH(128, 128, 16,  8, 16)
    DISPATCH(256, 256, 16, 16,  8)
    DISPATCH(256, 128, 24, 16,  8)
    DISPATCH(256, 128, 32, 16,  8)
    DISPATCH(128, 128, 24, 16,  8)
    DISPATCH(256, 256,  8, 16,  8)

    #undef DISPATCH

    // Unsupported config — throw so the benchmark harness (matmul.cpp:528 try-block)
    // can catch, report, and continue with other kernels instead of dying. A silent
    // return would leave cudaEvent timers with stale near-zero values and cause the
    // autotuner to falsely rank an unlaunched config as "best"; exit() would kill the
    // whole harness mid-sweep. Throwing is the right middle ground.
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "[Matmul2DBlocktileAuto] Unsupported config: BM=%d BN=%d BK=%d TM=%d TN=%d",
             BM, BN, BK, TM, TN);
    throw std::runtime_error(error_msg);
}

void Matmul2DBlocktileAuto::tune(const float *d_A, const float *d_B, float *d_C) {
    // RAII guard for the cudaEvent_t pair so we don't leak the events if
    // launch() throws (e.g. unsupported config or runtime error from a kernel
    // dispatch that didn't make it into the DISPATCH table). The guard creates
    // both events in its constructor and destroys them in its destructor — the
    // events are usable for the rest of this function via `start` / `stop`.
    struct EventGuard {
        cudaEvent_t &s, &e;
        EventGuard(cudaEvent_t &start_, cudaEvent_t &stop_) : s(start_), e(stop_) {
            cudaEventCreate(&s);
            cudaEventCreate(&e);
        }
        ~EventGuard() {
            cudaEventDestroy(s);
            cudaEventDestroy(e);
        }
    };
    cudaEvent_t start = nullptr, stop = nullptr;
    EventGuard guard(start, stop);

    int best_idx = -1;
    float best_ms = 1e30f;

    printf("[autotune 2d_blocktile N=%d] sweeping %d candidates...\n", N, NUM_CANDIDATES_2D);

    for (int i = 0; i < NUM_CANDIDATES_2D; i++) {
        Candidate2D c = CANDIDATES_2D[i];
        int BM = c.BM, BN = c.BN, BK = c.BK, TM = c.TM, TN = c.TN;
        int threads_per_block = (BM / TM) * (BN / TN);

        // Validity (mirrors kernel constexpr requirements):
        if (threads_per_block > 1024) continue;            // CUDA hard limit
        if (BM % TM != 0 || BN % TN != 0) continue;        // thread tile divides block tile
        if ((BM * BK) % threads_per_block != 0) continue;  // strided A load divides cleanly (BM)
        if ((BK * BN) % threads_per_block != 0) continue;  // strided B load divides cleanly (BK)
        // Stride-itself divisibility: strideA = NUM_THREADS/BK must divide BM, which
        // requires NUM_THREADS % BK == 0 (so integer-truncation of strideA doesn't
        // produce a stride that leaves uncovered rows / OOB writes into SMEM).
        // Same for strideB / BN.
        if (threads_per_block % BK != 0) continue;
        if (threads_per_block % BN != 0) continue;
        // SMEM check: (BM*BK + BK*BN) * sizeof(float) <= 48KB
        // Default static shared memory limit on CUDA is 48KB per block. H100 allows
        // up to 228KB per block via cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize),
        // but this kernel uses static __shared__ allocations and never opts in, so
        // 48KB is the hard limit.
        int smem_bytes = (BM * BK + BK * BN) * sizeof(float);
        if (smem_bytes > 48 * 1024) continue;
        // Boundary fairness: skip non-divisible N (autotune timing is biased by
        // partial-tile branches; benchmark sizes are powers of 2 anyway).
        if (N % BM != 0 || N % BN != 0) continue;

        // Clear any pre-existing CUDA error that might have leaked from prior
        // setup (cudaMalloc, prior candidate's edge cases, cudaEventCreate, etc.)
        // before we start warming up this candidate. This guarantees the
        // post-warmup cudaGetLastError() below catches only this candidate's errors.
        cudaGetLastError();

        // Warmup
        for (int w = 0; w < 2; w++) {
            launch(d_A, d_B, d_C, BM, BN, BK, TM, TN);
        }
        cudaDeviceSynchronize();

        // Check for launch failures (register spill, SMEM overrun, illegal access).
        // CUDA errors are sticky+async, so without this check a failed warmup leaves
        // stale near-zero event timings and the autotuner would falsely rank it best.
        // cudaGetLastError() also clears the sticky state for subsequent candidates.
        cudaError_t warmup_err = cudaGetLastError();
        if (warmup_err != cudaSuccess) {
            printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%2d TN=%2d  thr=%4d  ->  SKIPPED (%s)\n",
                   i, BM, BN, BK, TM, TN, threads_per_block, cudaGetErrorString(warmup_err));
            continue;
        }

        // Measure: 3 timed runs, median.
        // Initialize to +inf so that if a cudaEvent API fails (returns non-zero),
        // the uninitialized stack value can't be sorted as "fastest" and falsely
        // win the candidate ranking. Same defense as the post-warmup error check.
        float times[3] = {1e30f, 1e30f, 1e30f};
        bool timing_failed = false;
        for (int t = 0; t < 3; t++) {
            cudaEventRecord(start);
            launch(d_A, d_B, d_C, BM, BN, BK, TM, TN);
            cudaEventRecord(stop);
            if (cudaEventSynchronize(stop) != cudaSuccess ||
                cudaEventElapsedTime(&times[t], start, stop) != cudaSuccess) {
                timing_failed = true;
                break;
            }
        }
        if (timing_failed) {
            printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%2d TN=%2d  thr=%4d  ->  SKIPPED (timing failed)\n",
                   i, BM, BN, BK, TM, TN, threads_per_block);
            cudaGetLastError();  // clear so it doesn't leak into next candidate
            continue;
        }
        if (times[0] > times[1]) { float t = times[0]; times[0] = times[1]; times[1] = t; }
        if (times[1] > times[2]) { float t = times[1]; times[1] = times[2]; times[2] = t; }
        if (times[0] > times[1]) { float t = times[0]; times[0] = times[1]; times[1] = t; }
        float median = times[1];

        double tflops = (2.0 * (double)N * N * N) / (median * 1e9);
        printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%2d TN=%2d  thr=%4d smem=%2dKB  ->  %.3f ms  (%.2f TFLOPS)\n",
               i, BM, BN, BK, TM, TN, threads_per_block, smem_bytes / 1024, median, tflops);

        if (median < best_ms) {
            best_ms = median;
            best_idx = i;
        }
    }

    // Events are automatically destroyed by EventGuard when this function returns.

    if (best_idx < 0) {
        // All candidates skipped (e.g. odd N). Fall back to default.
        printf("[autotune 2d_blocktile N=%d] no candidate passed validity; falling back to default (128,128,8,8,8).\n", N);
        best_BM = 128; best_BN = 128; best_BK = 8; best_TM = 8; best_TN = 8;
        best_time_ms = 0.0f;
    } else {
        Candidate2D best = CANDIDATES_2D[best_idx];
        best_BM = best.BM;
        best_BN = best.BN;
        best_BK = best.BK;
        best_TM = best.TM;
        best_TN = best.TN;
        best_time_ms = best_ms;
        double best_tflops = (2.0 * (double)N * N * N) / (best_ms * 1e9);
        printf("[autotune 2d_blocktile N=%d] BEST: BM=%d BN=%d BK=%d TM=%d TN=%d  ->  %.3f ms  (%.2f TFLOPS)\n",
               N, best_BM, best_BN, best_BK, best_TM, best_TN, best_ms, best_tflops);
    }
    tuned = true;
}

Matmul2DBlocktileAuto::Matmul2DBlocktileAuto(int N, int blockDim)
    : N(N), blockDim(blockDim),
      best_BM(128), best_BN(128), best_BK(8), best_TM(8), best_TN(8),
      best_time_ms(0.0f), tuned(false) {}

void Matmul2DBlocktileAuto::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!tuned) {
        tune(d_A, d_B, d_C);
    }
    launch(d_A, d_B, d_C, best_BM, best_BN, best_BK, best_TM, best_TN);
    cudaCheckError(cudaGetLastError());
}

Matmul2DBlocktileAuto::~Matmul2DBlocktileAuto() {}
