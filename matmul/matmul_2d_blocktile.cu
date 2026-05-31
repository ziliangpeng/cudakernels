#include "matmul_2d_blocktile.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>

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
// 11-candidate sweep across (BM, BN, BK, TM, TN). Each candidate satisfies:
//   - NUM_THREADS = (BM/TM) * (BN/TN) <= 1024 (CUDA hard limit)
//   - NUM_THREADS divides BM*BK (A tile load)
//   - NUM_THREADS divides BK*BN (B tile load)
//   - 2 * (BM*BK + BK*BN) * 4 bytes <= 228KB SMEM (H100)
//   - TM*TN + TM + TN <= ~96 (rough register budget)
//
// The sweep itself catches any post-warmup CUDA errors and skips those configs.

struct Candidate2D {
    int BM, BN, BK, TM, TN;
};

static const Candidate2D CANDIDATES_2D[] = {
    // {BM, BN, BK, TM, TN}
    {128, 128,  8,  8,  8},   // [ 0] CURRENT DEFAULT — 256 threads, 64 acc, 16KB SMEM
    { 64,  64,  8,  8,  8},   //  [ 1] smaller block — 64 threads, more blocks/SM
    {128, 128,  8, 16,  8},   //  [ 2] bigger TM — 128 threads, 128 acc
    {128, 128,  8,  8, 16},   //  [ 3] bigger TN — 128 threads, 128 acc
    {128, 128, 16,  8,  8},   //  [ 4] deeper BK — 32KB SMEM, fewer K-iters
    {128,  64,  8,  8,  8},   //  [ 5] asymmetric (tall) — 128 threads
    { 64, 128,  8,  8,  8},   //  [ 6] asymmetric (wide) — 128 threads
    {128, 128,  8,  4,  4},   //  [ 7] small thread tile — 1024 threads, 16 acc
    {256, 128,  8, 16,  8},   //  [ 8] bigger block (tall) — 128 threads, 24KB SMEM
    {128, 256,  8,  8, 16},   //  [ 9] bigger block (wide) — 128 threads, 24KB SMEM
    {128, 128, 16, 16,  8},   // [10] deepest BK + bigger TM — 128 threads, 32KB SMEM
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

    #undef DISPATCH

    fprintf(stderr, "[Matmul2DBlocktileAuto] Unsupported config: BM=%d BN=%d BK=%d TM=%d TN=%d\n",
            BM, BN, BK, TM, TN);
}

void Matmul2DBlocktileAuto::tune(const float *d_A, const float *d_B, float *d_C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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
        if ((BM * BK) % threads_per_block != 0) continue;  // strided A load divides cleanly
        if ((BK * BN) % threads_per_block != 0) continue;  // strided B load divides cleanly
        // SMEM check: 2 * (BM*BK + BK*BN) * sizeof(float) <= 228KB
        int smem_bytes = 2 * (BM * BK + BK * BN) * sizeof(float);
        if (smem_bytes > 228 * 1024) continue;
        // Boundary fairness: skip non-divisible N (autotune timing is biased by
        // partial-tile branches; benchmark sizes are powers of 2 anyway).
        if (N % BM != 0 || N % BN != 0) continue;

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

        // Measure: 3 timed runs, median
        float times[3];
        for (int t = 0; t < 3; t++) {
            cudaEventRecord(start);
            launch(d_A, d_B, d_C, BM, BN, BK, TM, TN);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&times[t], start, stop);
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

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
