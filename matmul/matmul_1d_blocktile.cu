#include "matmul_1d_blocktile.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================================
// Templated 1D Block Tiling kernel
// ============================================================================
//
// Each thread computes TM elements in a column of C, increasing arithmetic
// intensity. Tile dimensions are template parameters so the same kernel source
// can be instantiated at many (BM, BN, BK, TM) shapes for autotuning.
//
// Identical algorithm to the original Matmul1DBlocktile, just parameterized.

template<int BM, int BN, int BK, int TM>
__global__ void matmul1DBlocktileKernelT(const float *A, const float *B, float *C, int N) {
    // Shared memory for tiles
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Thread indices
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // Block position in output
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Move pointers to the block's starting position
    A += blockRow * BM * N;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    // Thread results: TM partial sums per thread, held in registers.
    float threadResults[TM] = {0.0f};

    // Load indices for shared memory.
    // We have (BM*BN)/TM threads loading BM*BK A elements and BK*BN B elements.
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    // Loop over K dimension in chunks of BK
    for (int tileIdx = 0; tileIdx < N; tileIdx += BK) {
        // Load tile of A into shared memory
        if (blockRow * BM + innerRowA < N && tileIdx + innerColA < N) {
            As[innerRowA][innerColA] = A[innerRowA * N + innerColA];
        } else {
            As[innerRowA][innerColA] = 0.0f;
        }

        // Load tile of B into shared memory
        if (tileIdx + innerRowB < N && blockCol * BN + innerColB < N) {
            Bs[innerRowB][innerColB] = B[innerRowB * N + innerColB];
        } else {
            Bs[innerRowB][innerColB] = 0.0f;
        }

        __syncthreads();

        // Move pointers for next iteration
        A += BK;
        B += BK * N;

        // Compute TM elements
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            // Load B element once into a register, reuse it across TM A values.
            float tmpB = Bs[dotIdx][threadCol];
            #pragma unroll
            for (int resIdx = 0; resIdx < TM; resIdx++) {
                threadResults[resIdx] += As[threadRow * TM + resIdx][dotIdx] * tmpB;
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    #pragma unroll
    for (int resIdx = 0; resIdx < TM; resIdx++) {
        int globalRow = blockRow * BM + threadRow * TM + resIdx;
        int globalCol = blockCol * BN + threadCol;
        if (globalRow < N && globalCol < N) {
            C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
        }
    }
}

// ============================================================================
// Original (hardcoded) Matmul1DBlocktile — kept as baseline for comparison.
// Uses BM=64, BN=64, BK=8, TM=8 (matches siboehm Kernel 4 defaults).
// ============================================================================

#define BM_1D 64
#define BN_1D 64
#define BK_1D 8
#define TM_1D 8
#define NUM_THREADS_1D ((BM_1D / TM_1D) * BN_1D)

Matmul1DBlocktile::Matmul1DBlocktile(int N, int blockDim) : N(N), blockDim(blockDim) {}

void Matmul1DBlocktile::execute(const float *d_A, const float *d_B, float *d_C) {
    dim3 threads(NUM_THREADS_1D);
    dim3 blocks((N + BN_1D - 1) / BN_1D, (N + BM_1D - 1) / BM_1D);
    matmul1DBlocktileKernelT<BM_1D, BN_1D, BK_1D, TM_1D><<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());
}

Matmul1DBlocktile::~Matmul1DBlocktile() {}

// ============================================================================
// Autotuning version — Matmul1DBlocktileAuto
// ============================================================================
//
// Candidate grid: 13 (BM, BN, BK, TM) tuples chosen for 4K-friendly shapes.
// Each candidate:
//   - threads_per_block = (BM*BN)/TM must be <= 1024 (CUDA hard limit)
//   - BM and BN should divide common sizes (64..4096)
//   - TM controls register usage per thread (TM accumulators + 1 tmpB)
//
// The grid intentionally explores three axes:
//   - block tile size (BM, BN): 64..256
//   - K-loop chunk    (BK):    8 or 16
//   - thread tile     (TM):    4, 8, or 16

struct Candidate {
    int BM, BK, TM;
};

// IMPORTANT — constraint of the current kernel:
//   The kernel uses 1-element-per-thread load with
//     innerRowA = tid / BK, innerColA = tid % BK,  (covers BM*BK elements)
//     innerRowB = tid / BN, innerColB = tid % BN.  (covers BK*BN elements)
//   For correctness we therefore need:
//     NUM_THREADS == (BM*BN)/TM == BM*BK == BK*BN
//   which implies BM == BN and BM == BK*TM.
//
// Valid candidates satisfying this constraint:

static const Candidate CANDIDATES[] = {
    // {BM = BN, BK, TM}  threads = BM*BK
    { 32,  4,  8},   // [0] tiny, 256 threads
    { 32,  8,  4},   // [1] tiny, 256 threads (smaller TM = more outputs/block)
    { 64,  4, 16},   // [2] small block, large thread tile, 256 threads
    { 64, 16,  4},   // [3] small block, small thread tile, 256 threads
    { 64,  8,  8},   // [4] CURRENT DEFAULT (baseline), 512 threads
    {128,  8, 16},   // [5] big block, large thread tile, 1024 threads
    {128, 16,  8},   // [6] big block + deeper BK, 1024 threads
};
static const int NUM_CANDIDATES = sizeof(CANDIDATES) / sizeof(CANDIDATES[0]);

// Dispatch table — explicit instantiations.
// One branch per candidate. Order MUST match CANDIDATES[].
void Matmul1DBlocktileAuto::launch(const float *d_A, const float *d_B, float *d_C,
                                   int BM, int BN, int BK, int TM) {
    int threads_per_block = (BM * BN) / TM;
    dim3 threads(threads_per_block);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    #define DISPATCH(_BM, _BN, _BK, _TM) \
        if (BM == _BM && BN == _BN && BK == _BK && TM == _TM) { \
            matmul1DBlocktileKernelT<_BM, _BN, _BK, _TM><<<blocks, threads>>>(d_A, d_B, d_C, N); \
            return; \
        }

    DISPATCH( 32,  32,  4,  8)
    DISPATCH( 32,  32,  8,  4)
    DISPATCH( 64,  64,  4, 16)
    DISPATCH( 64,  64, 16,  4)
    DISPATCH( 64,  64,  8,  8)
    DISPATCH(128, 128,  8, 16)
    DISPATCH(128, 128, 16,  8)

    #undef DISPATCH

    fprintf(stderr, "[Matmul1DBlocktileAuto] Unsupported config: BM=%d BN=%d BK=%d TM=%d\n",
            BM, BN, BK, TM);
}

// Sweep candidates the first time execute() is called.
// Strategy: 2 warmup launches + 3 timed launches per candidate, pick median.
// Compare median times across candidates, pick the fastest.
void Matmul1DBlocktileAuto::tune(const float *d_A, const float *d_B, float *d_C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int best_idx = 0;
    float best_ms = 1e30f;

    printf("[autotune 1d_blocktile N=%d] sweeping %d candidates...\n", N, NUM_CANDIDATES);

    for (int i = 0; i < NUM_CANDIDATES; i++) {
        Candidate c = CANDIDATES[i];
        int BM = c.BM, BN = c.BM, BK = c.BK, TM = c.TM;  // BN == BM by constraint
        int threads_per_block = (BM * BN) / TM;
        if (threads_per_block > 1024) continue;
        if (BM % TM != 0) continue;
        if (N % BM != 0 || N % BN != 0) continue;

        for (int w = 0; w < 2; w++) {
            launch(d_A, d_B, d_C, BM, BN, BK, TM);
        }
        cudaDeviceSynchronize();

        float times[3];
        for (int t = 0; t < 3; t++) {
            cudaEventRecord(start);
            launch(d_A, d_B, d_C, BM, BN, BK, TM);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&times[t], start, stop);
        }
        if (times[0] > times[1]) { float t = times[0]; times[0] = times[1]; times[1] = t; }
        if (times[1] > times[2]) { float t = times[1]; times[1] = times[2]; times[2] = t; }
        if (times[0] > times[1]) { float t = times[0]; times[0] = times[1]; times[1] = t; }
        float median = times[1];

        double tflops = (2.0 * (double)N * N * N) / (median * 1e9);
        printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%2d  thr=%4d  ->  %.3f ms  (%.2f TFLOPS)\n",
               i, BM, BN, BK, TM, threads_per_block, median, tflops);

        if (median < best_ms) {
            best_ms = median;
            best_idx = i;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    Candidate best = CANDIDATES[best_idx];
    best_BM = best.BM;
    best_BN = best.BM;  // BN == BM by constraint
    best_BK = best.BK;
    best_TM = best.TM;
    best_time_ms = best_ms;
    tuned = true;

    double best_tflops = (2.0 * (double)N * N * N) / (best_ms * 1e9);
    printf("[autotune 1d_blocktile N=%d] BEST: BM=%d BN=%d BK=%d TM=%d  ->  %.3f ms  (%.2f TFLOPS)\n",
           N, best_BM, best_BN, best_BK, best_TM, best_ms, best_tflops);
}

Matmul1DBlocktileAuto::Matmul1DBlocktileAuto(int N, int blockDim)
    : N(N), blockDim(blockDim),
      best_BM(64), best_BN(64), best_BK(8), best_TM(8),
      best_time_ms(0.0f), tuned(false) {}

void Matmul1DBlocktileAuto::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!tuned) {
        tune(d_A, d_B, d_C);
    }
    launch(d_A, d_B, d_C, best_BM, best_BN, best_BK, best_TM);
    cudaCheckError(cudaGetLastError());
}

Matmul1DBlocktileAuto::~Matmul1DBlocktileAuto() {}
