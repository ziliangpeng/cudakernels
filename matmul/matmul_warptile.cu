#include "matmul_warptile.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

// ============================================================================
// Hardcoded Warp Tiling kernel (Kernel 10) — kept as baseline
// ============================================================================

#define BM_WARP 128
#define BN_WARP 128
#define BK_WARP 16
#define WM 64
#define WN 64
#define TM_WARP 8
#define TN_WARP 4
#define WARP_SIZE 32

#define WARPS_PER_BLOCK_X (BN_WARP / WN)
#define WARPS_PER_BLOCK_Y (BM_WARP / WM)
#define NUM_WARPS (WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y)

#define WARP_THREAD_M 4
#define WARP_THREAD_N 8
#define WARP_SUBTILE_M 2
#define WARP_SUBTILE_N 2

#define NUM_THREADS_WARP (NUM_WARPS * WARP_SIZE)

__global__ void matmulWarptileKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float As[BK_WARP][BM_WARP];
    __shared__ float Bs[BK_WARP][BN_WARP];

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpRow = warpId / WARPS_PER_BLOCK_X;
    const int warpCol = warpId % WARPS_PER_BLOCK_X;
    const int threadRowInWarp = laneId / WARP_THREAD_N;
    const int threadColInWarp = laneId % WARP_THREAD_N;
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += blockRow * BM_WARP * N;
    B += blockCol * BN_WARP;
    C += blockRow * BM_WARP * N + blockCol * BN_WARP;

    float threadResults[WARP_SUBTILE_M * TM_WARP][WARP_SUBTILE_N * TN_WARP] = {{0.0f}};
    float regA[WARP_SUBTILE_M * TM_WARP];
    float regB[WARP_SUBTILE_N * TN_WARP];

    const int strideA = NUM_THREADS_WARP / BK_WARP;
    const int strideB = NUM_THREADS_WARP / BN_WARP;
    const int innerRowA = threadIdx.x / BK_WARP;
    const int innerColA = threadIdx.x % BK_WARP;
    const int innerRowB = threadIdx.x / BN_WARP;
    const int innerColB = threadIdx.x % BN_WARP;

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_WARP) {
        for (int loadOffset = 0; loadOffset < BM_WARP; loadOffset += strideA) {
            int row = innerRowA + loadOffset;
            if (blockRow * BM_WARP + row < N && tileIdx + innerColA < N)
                As[innerColA][row] = A[row * N + innerColA];
            else
                As[innerColA][row] = 0.0f;
        }
        for (int loadOffset = 0; loadOffset < BK_WARP; loadOffset += strideB) {
            int row = innerRowB + loadOffset;
            if (tileIdx + row < N && blockCol * BN_WARP + innerColB < N)
                Bs[row][innerColB] = B[row * N + innerColB];
            else
                Bs[row][innerColB] = 0.0f;
        }
        __syncthreads();
        A += BK_WARP;
        B += BK_WARP * N;

        for (int dotIdx = 0; dotIdx < BK_WARP; dotIdx++) {
            for (int subtileM = 0; subtileM < WARP_SUBTILE_M; subtileM++)
                for (int i = 0; i < TM_WARP; i++)
                    regA[subtileM * TM_WARP + i] = As[dotIdx][warpRow * WM + subtileM * (WM / WARP_SUBTILE_M) + threadRowInWarp * TM_WARP + i];
            for (int subtileN = 0; subtileN < WARP_SUBTILE_N; subtileN++)
                for (int j = 0; j < TN_WARP; j++)
                    regB[subtileN * TN_WARP + j] = Bs[dotIdx][warpCol * WN + subtileN * (WN / WARP_SUBTILE_N) + threadColInWarp * TN_WARP + j];
            for (int i = 0; i < WARP_SUBTILE_M * TM_WARP; i++)
                for (int j = 0; j < WARP_SUBTILE_N * TN_WARP; j++)
                    threadResults[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

    for (int subtileM = 0; subtileM < WARP_SUBTILE_M; subtileM++)
        for (int i = 0; i < TM_WARP; i++) {
            int globalRow = blockRow * BM_WARP + warpRow * WM + subtileM * (WM / WARP_SUBTILE_M) + threadRowInWarp * TM_WARP + i;
            if (globalRow < N)
                for (int subtileN = 0; subtileN < WARP_SUBTILE_N; subtileN++)
                    for (int j = 0; j < TN_WARP; j++) {
                        int globalCol = blockCol * BN_WARP + warpCol * WN + subtileN * (WN / WARP_SUBTILE_N) + threadColInWarp * TN_WARP + j;
                        if (globalCol < N)
                            C[(warpRow * WM + subtileM * (WM / WARP_SUBTILE_M) + threadRowInWarp * TM_WARP + i) * N +
                              (warpCol * WN + subtileN * (WN / WARP_SUBTILE_N) + threadColInWarp * TN_WARP + j)] =
                                threadResults[subtileM * TM_WARP + i][subtileN * TN_WARP + j];
                    }
        }
}

MatmulWarptile::MatmulWarptile(int N, int blockDim) : N(N), blockDim(blockDim) {}
void MatmulWarptile::execute(const float *d_A, const float *d_B, float *d_C) {
    dim3 threads(NUM_THREADS_WARP);
    dim3 blocks((N + BN_WARP - 1) / BN_WARP, (N + BM_WARP - 1) / BM_WARP);
    matmulWarptileKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());
}
MatmulWarptile::~MatmulWarptile() {}

// Undefine all hardcoded macros so they don't collide with autotune section
#undef BM_WARP
#undef BN_WARP
#undef BK_WARP
#undef WM
#undef WN
#undef TM_WARP
#undef TN_WARP
#undef WARP_SIZE
#undef WARPS_PER_BLOCK_X
#undef WARPS_PER_BLOCK_Y
#undef NUM_WARPS
#undef WARP_THREAD_M
#undef WARP_THREAD_N
#undef WARP_SUBTILE_M
#undef WARP_SUBTILE_N
#undef NUM_THREADS_WARP

// ============================================================================
// Templated Warp Tiling kernel (autotune version)
// ============================================================================
//
// Three-level hierarchy: Block -> Warp -> Thread
//   - Block tile: BM x BN
//   - Warp tile:  WM x WN   (each warp computes one WM x WN region)
//   - Thread tile: TM x TN  (each thread computes one or more TM x TN subtiles)
//
// Warp layout (fixed for all candidates):
//   - 32 threads per warp, arranged as WARP_THREAD_M=4 in M, WARP_THREAD_N=8 in N
//   - Each thread covers WM/(4*TM) * WN/(8*TN) subtiles (>= 1)
//
// SMEM: transposed As[BK][BM], Bs[BK][BN]. Strided loads like 2D blocktile.
// C output: thread-local accumulator, written as local offset from warp origin.

template<int BM, int BN, int BK, int TM, int TN, int WARP_M, int WARP_N>
__global__ void matmulWarptileKernelT(const float * __restrict__ A,
                                       const float * __restrict__ B,
                                       float *C, int N) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    constexpr int WARP_SIZE = 32;
    constexpr int WARP_THREAD_M = 4;
    constexpr int WARP_THREAD_N = 8;
    static_assert(WARP_THREAD_M * WARP_THREAD_N == WARP_SIZE);

    // Number of warps per block
    constexpr int WARPS_X = BN / WARP_N;
    constexpr int WARPS_Y = BM / WARP_M;
    constexpr int NUM_WARPS = WARPS_X * WARPS_Y;
    constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

    // Subtiles per thread within a warp
    constexpr int STM = WARP_M / (WARP_THREAD_M * TM);
    constexpr int STN = WARP_N / (WARP_THREAD_N * TN);
    static_assert(STM >= 1 && STN >= 1);

    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpRow = warpId / WARPS_X;
    const int warpCol = warpId % WARPS_X;
    const int thrRow = laneId / WARP_THREAD_N;
    const int thrCol = laneId % WARP_THREAD_N;
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += blockRow * BM * N;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    float acc[STM * TM][STN * TN];
    #pragma unroll
    for (int i = 0; i < STM * TM; i++)
        #pragma unroll
        for (int j = 0; j < STN * TN; j++)
            acc[i][j] = 0.0f;

    float regA[STM * TM];
    float regB[STN * TN];

    // Strided load: each thread loads multiple elements from A and B tiles
    constexpr int strideA = NUM_THREADS / BK;
    constexpr int strideB = NUM_THREADS / BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    for (int tileK = 0; tileK < N; tileK += BK) {
        // Load A tile (transposed: As[BK][BM])
        #pragma unroll
        for (int off = 0; off < BM; off += strideA) {
            int r = innerRowA + off;
            if (blockRow * BM + r < N && tileK + innerColA < N)
                As[innerColA][r] = A[r * N + innerColA];
            else
                As[innerColA][r] = 0.0f;
        }
        // Load B tile
        #pragma unroll
        for (int off = 0; off < BK; off += strideB) {
            int r = innerRowB + off;
            if (tileK + r < N && blockCol * BN + innerColB < N)
                Bs[r][innerColB] = B[r * N + innerColB];
            else
                Bs[r][innerColB] = 0.0f;
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        #pragma unroll
        for (int d = 0; d < BK; d++) {
            // regA: load all subtiles in M
            #pragma unroll
            for (int sm = 0; sm < STM; sm++)
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    regA[sm * TM + i] = As[d][warpRow * WARP_M + sm * (WARP_M / STM) + thrRow * TM + i];

            // regB: load all subtiles in N
            #pragma unroll
            for (int sn = 0; sn < STN; sn++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    regB[sn * TN + j] = Bs[d][warpCol * WARP_N + sn * (WARP_N / STN) + thrCol * TN + j];

            // Outer product over all subtiles
            #pragma unroll
            for (int i = 0; i < STM * TM; i++)
                #pragma unroll
                for (int j = 0; j < STN * TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int sm = 0; sm < STM; sm++)
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            int gr = blockRow * BM + warpRow * WARP_M + sm * (WARP_M / STM) + thrRow * TM + i;
            if (gr < N) {
                #pragma unroll
                for (int sn = 0; sn < STN; sn++)
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        int gc = blockCol * BN + warpCol * WARP_N + sn * (WARP_N / STN) + thrCol * TN + j;
                        if (gc < N)
                            C[(warpRow * WARP_M + sm * (WARP_M / STM) + thrRow * TM + i) * N +
                              (warpCol * WARP_N + sn * (WARP_N / STN) + thrCol * TN + j)] =
                                acc[sm * TM + i][sn * TN + j];
                    }
            }
        }
}

// ============================================================================
// Autotuning — MatmulWarptileAuto
// ============================================================================

struct CandidateW {
    int BM, BN, BK, TM, TN, WM, WN;
};

static const CandidateW CANDIDATES_W[] = {
    // {BM,  BN,  BK, TM, TN, WM, WN}  SMEM = (BM*BK + BK*BN)*4 bytes
    {128, 128, 16,  8,  4, 64,  64},   // [ 0] default-like — 4 warps, 16KB SMEM
    {128, 128, 16, 16,  8, 64,  64},   // [ 1] bigger thread tile — same 4 warps
    {128, 128, 16,  8,  8, 64,  64},   // [ 2] square thread tile — 4 warps
    {128, 128,  8, 16,  8, 64,  64},   // [ 3] shallower BK — 8KB SMEM
    {128, 128, 16, 16,  8, 64, 128},   // [ 4] wider warp — 2 warps, BN/2 per warp
    {128, 128, 16, 16,  8,128,  64},   // [ 5] taller warp — 2 warps, BM/2 per warp
    {256, 128, 16, 16,  8, 64,  64},   // [ 6] taller block — 8 warps, 24KB SMEM
    {128, 256, 16, 16,  8, 64,  64},   // [ 7] wider block — 8 warps, 24KB SMEM
    {128, 128,  8,  8,  4, 64,  64},   // [ 8] small thread tile (like default)
    {128, 128,  8, 16,  4, 64,  64},   // [ 9] BK=8, TM=16, square-tiled M
    {128, 128, 16,  8,  4, 32,  32},   // [10] small warp tile — 16 warps
    {128, 128, 16,  8,  4, 32,  64},   // [11] small warp M × medium warp N
    {128, 128, 16,  8,  4, 64,  32},   // [12] medium warp M × small warp N
    {128, 128,  8,  4,  4, 64,  64},   // [13] tiny thread tile — 4 warps, 1024 thr
    {256, 128, 16, 16,  8,128, 128},   // [14] big warp, tall block — 2 warps
    {128, 128, 16,  8,  8, 64, 128},   // [15] wide warp, square thread — 2 warps
    {128, 128, 16,  8,  8,128,  64},   // [16] tall warp, square thread — 2 warps
};
static const int NUM_CANDIDATES_W = sizeof(CANDIDATES_W) / sizeof(CANDIDATES_W[0]);

void MatmulWarptileAuto::launch(const float *d_A, const float *d_B, float *d_C,
                                int BM, int BN, int BK, int TM, int TN, int WM, int WN) {
    int warps_x = BN / WN;
    int warps_y = BM / WM;
    int num_warps = warps_x * warps_y;
    int threads_per_block = num_warps * 32;
    dim3 threads(threads_per_block);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    #define DISPATCH(_BM, _BN, _BK, _TM, _TN, _WM, _WN) \
        if (BM == _BM && BN == _BN && BK == _BK && TM == _TM && TN == _TN && WM == _WM && WN == _WN) { \
            matmulWarptileKernelT<_BM, _BN, _BK, _TM, _TN, _WM, _WN><<<blocks, threads>>>(d_A, d_B, d_C, N); \
            return; \
        }

    DISPATCH(128, 128, 16,  8,  4, 64,  64)
    DISPATCH(128, 128, 16, 16,  8, 64,  64)
    DISPATCH(128, 128, 16,  8,  8, 64,  64)
    DISPATCH(128, 128,  8, 16,  8, 64,  64)
    DISPATCH(128, 128, 16, 16,  8, 64, 128)
    DISPATCH(128, 128, 16, 16,  8,128,  64)
    DISPATCH(256, 128, 16, 16,  8, 64,  64)
    DISPATCH(128, 256, 16, 16,  8, 64,  64)
    DISPATCH(128, 128,  8,  8,  4, 64,  64)
    DISPATCH(128, 128,  8, 16,  4, 64,  64)
    DISPATCH(128, 128, 16,  8,  4, 32,  32)
    DISPATCH(128, 128, 16,  8,  4, 32,  64)
    DISPATCH(128, 128, 16,  8,  4, 64,  32)
    DISPATCH(128, 128,  8,  4,  4, 64,  64)
    DISPATCH(256, 128, 16, 16,  8,128, 128)
    DISPATCH(128, 128, 16,  8,  8, 64, 128)
    DISPATCH(128, 128, 16,  8,  8,128,  64)

    #undef DISPATCH

    char err[256];
    snprintf(err, sizeof(err),
             "[MatmulWarptileAuto] Unsupported config: BM=%d BN=%d BK=%d TM=%d TN=%d WM=%d WN=%d",
             BM, BN, BK, TM, TN, WM, WN);
    throw std::runtime_error(err);
}

void MatmulWarptileAuto::tune(const float *d_A, const float *d_B, float *d_C) {
    struct EventGuard {
        cudaEvent_t &s, &e;
        EventGuard(cudaEvent_t &start_, cudaEvent_t &stop_) : s(start_), e(stop_) {
            if (cudaEventCreate(&s) != cudaSuccess)
                throw std::runtime_error("EventGuard: cudaEventCreate failed for start");
            if (cudaEventCreate(&e) != cudaSuccess) {
                cudaEventDestroy(s);
                throw std::runtime_error("EventGuard: cudaEventCreate failed for stop");
            }
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

    printf("[autotune warptile N=%d] sweeping %d candidates...\n", N, NUM_CANDIDATES_W);

    for (int i = 0; i < NUM_CANDIDATES_W; i++) {
        CandidateW c = CANDIDATES_W[i];
        int BM = c.BM, BN = c.BN, BK = c.BK, TM = c.TM, TN = c.TN, WM = c.WM, WN = c.WN;

        // Validity checks
        int warps_x = BN / WN;
        int warps_y = BM / WM;
        int num_warps = warps_x * warps_y;
        int threads_per_block = num_warps * 32;
        if (threads_per_block > 1024) continue;
        if (BN % WN != 0 || BM % WM != 0) continue;
        if (WM % TM != 0 || WN % TN != 0) continue;
        // Subtiles must be integer
        int stm = WM / (4 * TM);
        int stn = WN / (8 * TN);
        if (stm < 1 || stn < 1) continue;
        if (WM % (4 * TM) != 0 || WN % (8 * TN) != 0) continue;
        // SMEM
        int smem_bytes = (BM * BK + BK * BN) * sizeof(float);
        if (smem_bytes > 48 * 1024) continue;
        // Divisibility for strided load
        if ((BM * BK) % threads_per_block != 0) continue;
        if ((BK * BN) % threads_per_block != 0) continue;
        if (threads_per_block % BK != 0) continue;
        if (threads_per_block % BN != 0) continue;
        if (N % BM != 0 || N % BN != 0) continue;

        cudaGetLastError();

        try {
            for (int w = 0; w < 2; w++)
                launch(d_A, d_B, d_C, BM, BN, BK, TM, TN, WM, WN);
        } catch (const std::exception &ex) {
            printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%2d TN=%2d WM=%3d WN=%3d  thr=%4d  ->  SKIPPED (launch: %s)\n",
                   i, BM, BN, BK, TM, TN, WM, WN, threads_per_block, ex.what());
            cudaGetLastError();
            continue;
        }
        cudaDeviceSynchronize();
        cudaError_t warmup_err = cudaGetLastError();
        if (warmup_err != cudaSuccess) {
            printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%2d TN=%2d WM=%3d WN=%3d  thr=%4d  ->  SKIPPED (%s)\n",
                   i, BM, BN, BK, TM, TN, WM, WN, threads_per_block, cudaGetErrorString(warmup_err));
            continue;
        }

        float times[3] = {1e30f, 1e30f, 1e30f};
        bool failed = false;
        for (int t = 0; t < 3; t++) {
            cudaEventRecord(start);
            launch(d_A, d_B, d_C, BM, BN, BK, TM, TN, WM, WN);
            cudaEventRecord(stop);
            if (cudaEventSynchronize(stop) != cudaSuccess ||
                cudaEventElapsedTime(&times[t], start, stop) != cudaSuccess) {
                failed = true;
                break;
            }
        }
        if (failed) {
            printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%2d TN=%2d WM=%3d WN=%3d  thr=%4d  ->  SKIPPED (timing)\n",
                   i, BM, BN, BK, TM, TN, WM, WN, threads_per_block);
            cudaGetLastError();
            continue;
        }
        if (times[0] > times[1]) { float t = times[0]; times[0] = times[1]; times[1] = t; }
        if (times[1] > times[2]) { float t = times[1]; times[1] = times[2]; times[2] = t; }
        if (times[0] > times[1]) { float t = times[0]; times[0] = times[1]; times[1] = t; }
        float median = times[1];

        double tflops = (2.0 * (double)N * N * N) / (median * 1e9);
        printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%2d TN=%2d WM=%3d WN=%3d  thr=%4d smem=%2dKB  ->  %.3f ms  (%.2f TFLOPS)\n",
               i, BM, BN, BK, TM, TN, WM, WN, threads_per_block, smem_bytes / 1024, median, tflops);

        if (median < best_ms) {
            best_ms = median;
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        printf("[autotune warptile N=%d] no valid candidate; fallback to default (128,128,16,8,4,64,64).\n", N);
        best_BM = 128; best_BN = 128; best_BK = 16; best_TM = 8; best_TN = 4; best_WM = 64; best_WN = 64;
        best_time_ms = 0.0f;
    } else {
        CandidateW best = CANDIDATES_W[best_idx];
        best_BM = best.BM; best_BN = best.BN; best_BK = best.BK;
        best_TM = best.TM; best_TN = best.TN; best_WM = best.WM; best_WN = best.WN;
        best_time_ms = best_ms;
        double tflops = (2.0 * (double)N * N * N) / (best_ms * 1e9);
        printf("[autotune warptile N=%d] BEST: BM=%d BN=%d BK=%d TM=%d TN=%d WM=%d WN=%d  ->  %.3f ms  (%.2f TFLOPS)\n",
               N, best_BM, best_BN, best_BK, best_TM, best_TN, best_WM, best_WN, best_ms, tflops);
    }
    tuned = true;
}

MatmulWarptileAuto::MatmulWarptileAuto(int N, int blockDim)
    : N(N), blockDim(blockDim),
      best_BM(128), best_BN(128), best_BK(16), best_TM(8), best_TN(4),
      best_WM(64), best_WN(64),
      best_time_ms(0.0f), tuned(false) {}

void MatmulWarptileAuto::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!tuned)
        tune(d_A, d_B, d_C);
    launch(d_A, d_B, d_C, best_BM, best_BN, best_BK, best_TM, best_TN, best_WM, best_WN);
    cudaCheckError(cudaGetLastError());
}

MatmulWarptileAuto::~MatmulWarptileAuto() {}
