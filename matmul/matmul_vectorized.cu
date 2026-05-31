#include "matmul_vectorized.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

// ============================================================================
// Templated Vectorized kernel — 2D blocktile + float4 C stores
// ============================================================================
//
// Same structure as 2D blocktile (non-transposed As[BM][BK], strided scalar
// GMEM loads, outer product compute) PLUS float4 (128-bit) stores for C output.
//
// The float4 store exploits H100's 128-byte L1 cache sector: 8 consecutive
// float writes from the same warp → fewer L1 transactions for the C output
// phase.
//
// Boundary handling: fallback to scalar stores when the last partial column
// of C doesn't have 4+ elements remaining.

template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmulVectorizedKernelT(const float * __restrict__ A,
                                        const float * __restrict__ B,
                                        float *C, int N) {
    // Non-transposed As[BM][BK] — same layout as 2D blocktile winner
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
        // Load A tile (strided scalar) — same as 2D blocktile
        #pragma unroll
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            int row = innerRowA + loadOffset;  // rows 0..BM-1
            if (blockRow * BM + row < N && tileIdx + innerColA < N) {
                As[row][innerColA] = A[row * N + innerColA];
            } else {
                As[row][innerColA] = 0.0f;
            }
        }

        // Load B tile (strided scalar) — same as 2D blocktile
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

        // Outer product compute — same as 2D blocktile
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

    // Write results using float4 (128-bit) stores — the key optimization
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int globalRow = blockRow * BM + threadRow * TM + i;
        if (globalRow < N) {
            #pragma unroll
            for (int j = 0; j < TN; j += 4) {
                int globalCol = blockCol * BN + threadCol * TN + j;
                if (globalCol + 3 < N) {
                    float4 tmp;
                    tmp.x = threadResults[i][j + 0];
                    tmp.y = threadResults[i][j + 1];
                    tmp.z = threadResults[i][j + 2];
                    tmp.w = threadResults[i][j + 3];
                    *reinterpret_cast<float4*>(
                        &C[(threadRow * TM + i) * N + threadCol * TN + j]) = tmp;
                } else {
                    #pragma unroll
                    for (int k = 0; k < 4 && globalCol + k < N; k++) {
                        C[(threadRow * TM + i) * N + threadCol * TN + j + k]
                            = threadResults[i][j + k];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Original (hardcoded) MatmulVectorized — kept as baseline.
// Uses float4 GMEM loads + transposed As + float4 C stores.
// BM=BN=128, BK=8, TM=TN=8.
// ============================================================================

#define BM_VEC 128
#define BN_VEC 128
#define BK_VEC 8
#define TM_VEC 8
#define TN_VEC 8
#define NUM_THREADS_VEC ((BM_VEC / TM_VEC) * (BN_VEC / TN_VEC))

__global__ void matmulVectorizedKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float As[BK_VEC][BM_VEC];
    __shared__ float Bs[BK_VEC][BN_VEC];

    const int threadCol = threadIdx.x % (BN_VEC / TN_VEC);
    const int threadRow = threadIdx.x / (BN_VEC / TN_VEC);

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    A += blockRow * BM_VEC * N;
    B += blockCol * BN_VEC;
    C += blockRow * BM_VEC * N + blockCol * BN_VEC;

    float threadResults[TM_VEC][TN_VEC] = {{0.0f}};
    float regA[TM_VEC];
    float regB[TN_VEC];

    const int innerRowA = threadIdx.x / (BK_VEC / 4);
    const int innerColA = threadIdx.x % (BK_VEC / 4);

    const int innerRowB = threadIdx.x / (BN_VEC / 4);
    const int innerColB = threadIdx.x % (BN_VEC / 4);

    for (int tileIdx = 0; tileIdx < N; tileIdx += BK_VEC) {
        // Load A with float4, store transposed into As[BK][BM]
        if (innerRowA < BM_VEC && blockRow * BM_VEC + innerRowA < N) {
            float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};
            if (tileIdx + innerColA * 4 + 3 < N) {
                tmp = *reinterpret_cast<const float4*>(&A[innerRowA * N + innerColA * 4]);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (tileIdx + innerColA * 4 + i < N) {
                        reinterpret_cast<float*>(&tmp)[i] = A[innerRowA * N + innerColA * 4 + i];
                    }
                }
            }
            As[innerColA * 4 + 0][innerRowA] = tmp.x;
            As[innerColA * 4 + 1][innerRowA] = tmp.y;
            As[innerColA * 4 + 2][innerRowA] = tmp.z;
            As[innerColA * 4 + 3][innerRowA] = tmp.w;
        }

        // Load B with float4
        if (innerRowB < BK_VEC && tileIdx + innerRowB < N) {
            float4 tmp = {0.0f, 0.0f, 0.0f, 0.0f};
            if (blockCol * BN_VEC + innerColB * 4 + 3 < N) {
                tmp = *reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4]);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (blockCol * BN_VEC + innerColB * 4 + i < N) {
                        reinterpret_cast<float*>(&tmp)[i] = B[innerRowB * N + innerColB * 4 + i];
                    }
                }
            }
            Bs[innerRowB][innerColB * 4 + 0] = tmp.x;
            Bs[innerRowB][innerColB * 4 + 1] = tmp.y;
            Bs[innerRowB][innerColB * 4 + 2] = tmp.z;
            Bs[innerRowB][innerColB * 4 + 3] = tmp.w;
        }

        __syncthreads();

        A += BK_VEC;
        B += BK_VEC * N;

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK_VEC; dotIdx++) {
            #pragma unroll
            for (int i = 0; i < TM_VEC; i++) {
                regA[i] = As[dotIdx][threadRow * TM_VEC + i];
            }
            #pragma unroll
            for (int j = 0; j < TN_VEC; j++) {
                regB[j] = Bs[dotIdx][threadCol * TN_VEC + j];
            }
            #pragma unroll
            for (int i = 0; i < TM_VEC; i++) {
                #pragma unroll
                for (int j = 0; j < TN_VEC; j++) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results using float4
    #pragma unroll
    for (int i = 0; i < TM_VEC; i++) {
        int globalRow = blockRow * BM_VEC + threadRow * TM_VEC + i;
        if (globalRow < N) {
            #pragma unroll
            for (int j = 0; j < TN_VEC; j += 4) {
                int globalCol = blockCol * BN_VEC + threadCol * TN_VEC + j;
                if (globalCol + 3 < N) {
                    float4 tmp;
                    tmp.x = threadResults[i][j + 0];
                    tmp.y = threadResults[i][j + 1];
                    tmp.z = threadResults[i][j + 2];
                    tmp.w = threadResults[i][j + 3];
                    *reinterpret_cast<float4*>(
                        &C[(threadRow * TM_VEC + i) * N + threadCol * TN_VEC + j]) = tmp;
                } else {
                    for (int k = 0; k < 4 && globalCol + k < N; k++) {
                        C[(threadRow * TM_VEC + i) * N + threadCol * TN_VEC + j + k]
                            = threadResults[i][j + k];
                    }
                }
            }
        }
    }
}

MatmulVectorized::MatmulVectorized(int N, int blockDim) : N(N), blockDim(blockDim) {}

void MatmulVectorized::execute(const float *d_A, const float *d_B, float *d_C) {
    dim3 threads(NUM_THREADS_VEC);
    dim3 blocks((N + BN_VEC - 1) / BN_VEC,
                (N + BM_VEC - 1) / BM_VEC);
    matmulVectorizedKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());
}

MatmulVectorized::~MatmulVectorized() {}

// ============================================================================
// Autotuning version — MatmulVectorizedAuto
// ============================================================================
//
// Uses 2D blocktile structure (non-transposed As[BM][BK], strided scalar
// GMEM loads) + float4 C stores. Same ~15-candidate sweep as 2D autotune
// with TN % 4 == 0 constraint for float4 store alignment.
//
// This directly answers: does float4 C store add anything on top of the 2D
// blocktile winner's scalar stores?

struct CandidateVec {
    int BM, BN, BK, TM, TN;
};

static const CandidateVec CANDIDATES_VEC[] = {
    // {BM, BN, BK, TM, TN}  SMEM = (BM*BK + BK*BN)*4 bytes
    {128, 128,  8,  8,  8},   // [ 0] DEFAULT — same as hardcoded
    {128, 128,  8, 16,  8},   // [ 1] bigger TM
    {128, 128,  8,  8, 16},   // [ 2] bigger TN
    {128, 128, 16,  8,  8},   // [ 3] deeper BK
    {128, 128, 16, 16,  8},   // [ 4] BK=16 + big TM (2D winner)
    {128, 128, 16,  8, 16},   // [ 5] BK=16 + big TN
    {128,  64,  8, 16,  8},   // [ 6] asymmetric (tall) — 128 thr, 6KB
    { 64, 128,  8,  8, 16},   // [ 7] asymmetric (wide) — 128 thr, 6KB
    {256, 128,  8, 16,  8},   // [ 8] bigger block (tall) — 256 thr, 12KB
    {128, 256,  8,  8, 16},   // [ 9] bigger block (wide) — 256 thr, 12KB
    { 64,  64,  8,  8,  8},   // [10] smaller block — 64 thr, 4KB
    { 64,  64, 16,  8,  8},   // [11] smaller + deeper BK — 64 thr, 8KB
    {128, 128, 32, 16,  8},   // [12] deeper BK on winner — 32KB
    {256, 128, 16, 16,  8},   // [13] big tall + BK=16 — 24KB
    {256, 256,  8, 16,  8},   // [14] big square — 16KB (may spill)
    {128, 128, 16,  4,  4},   // [15] small thread tile — 1024 thr, 16KB
};
static const int NUM_CANDIDATES_VEC = sizeof(CANDIDATES_VEC) / sizeof(CANDIDATES_VEC[0]);

void MatmulVectorizedAuto::launch(const float *d_A, const float *d_B, float *d_C,
                                  int BM, int BN, int BK, int TM, int TN) {
    int threads_per_block = (BM / TM) * (BN / TN);
    dim3 threads(threads_per_block);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);

    #define DISPATCH(_BM, _BN, _BK, _TM, _TN) \
        if (BM == _BM && BN == _BN && BK == _BK && TM == _TM && TN == _TN) { \
            matmulVectorizedKernelT<_BM, _BN, _BK, _TM, _TN><<<blocks, threads>>>(d_A, d_B, d_C, N); \
            return; \
        }

    DISPATCH(128, 128,  8,  8,  8)
    DISPATCH(128, 128,  8, 16,  8)
    DISPATCH(128, 128,  8,  8, 16)
    DISPATCH(128, 128, 16,  8,  8)
    DISPATCH(128, 128, 16, 16,  8)
    DISPATCH(128, 128, 16,  8, 16)
    DISPATCH(128,  64,  8, 16,  8)
    DISPATCH( 64, 128,  8,  8, 16)
    DISPATCH(256, 128,  8, 16,  8)
    DISPATCH(128, 256,  8,  8, 16)
    DISPATCH( 64,  64,  8,  8,  8)
    DISPATCH( 64,  64, 16,  8,  8)
    DISPATCH(128, 128, 32, 16,  8)
    DISPATCH(256, 128, 16, 16,  8)
    DISPATCH(256, 256,  8, 16,  8)
    DISPATCH(128, 128, 16,  4,  4)

    #undef DISPATCH

    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg),
             "[MatmulVectorizedAuto] Unsupported config: BM=%d BN=%d BK=%d TM=%d TN=%d",
             BM, BN, BK, TM, TN);
    throw std::runtime_error(error_msg);
}

void MatmulVectorizedAuto::tune(const float *d_A, const float *d_B, float *d_C) {
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

    printf("[autotune vectorized N=%d] sweeping %d candidates...\n", N, NUM_CANDIDATES_VEC);

    for (int i = 0; i < NUM_CANDIDATES_VEC; i++) {
        CandidateVec c = CANDIDATES_VEC[i];
        int BM = c.BM, BN = c.BN, BK = c.BK, TM = c.TM, TN = c.TN;
        int threads_per_block = (BM / TM) * (BN / TN);

        // Validity checks (same as 2D blocktile + TN%4==0 for float4 stores)
        if (threads_per_block > 1024) continue;
        if (BM % TM != 0 || BN % TN != 0) continue;
        if ((BM * BK) % threads_per_block != 0) continue;
        if ((BK * BN) % threads_per_block != 0) continue;
        if (threads_per_block % BK != 0) continue;
        if (threads_per_block % BN != 0) continue;
        if (TN % 4 != 0) continue;  // float4 store alignment
        int smem_bytes = (BM * BK + BK * BN) * sizeof(float);
        if (smem_bytes > 48 * 1024) continue;
        if (N % BM != 0 || N % BN != 0) continue;

        cudaGetLastError();

        try {
            for (int w = 0; w < 2; w++)
                launch(d_A, d_B, d_C, BM, BN, BK, TM, TN);
        } catch (const std::exception &ex) {
            printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%2d TN=%2d  thr=%4d  ->  SKIPPED (launch threw: %s)\n",
                   i, BM, BN, BK, TM, TN, threads_per_block, ex.what());
            cudaGetLastError();
            continue;
        }
        cudaDeviceSynchronize();
        cudaError_t warmup_err = cudaGetLastError();
        if (warmup_err != cudaSuccess) {
            printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%2d TN=%2d  thr=%4d  ->  SKIPPED (%s)\n",
                   i, BM, BN, BK, TM, TN, threads_per_block, cudaGetErrorString(warmup_err));
            continue;
        }

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
            cudaGetLastError();
            continue;
        }
        // Sort to find median
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

    if (best_idx < 0) {
        printf("[autotune vectorized N=%d] no candidate passed validity; fallback to (128,128,8,8,8).\n", N);
        best_BM = 128; best_BN = 128; best_BK = 8; best_TM = 8; best_TN = 8;
        best_time_ms = 0.0f;
    } else {
        CandidateVec best = CANDIDATES_VEC[best_idx];
        best_BM = best.BM;
        best_BN = best.BN;
        best_BK = best.BK;
        best_TM = best.TM;
        best_TN = best.TN;
        best_time_ms = best_ms;
        double best_tflops = (2.0 * (double)N * N * N) / (best_ms * 1e9);
        printf("[autotune vectorized N=%d] BEST: BM=%d BN=%d BK=%d TM=%d TN=%d  ->  %.3f ms  (%.2f TFLOPS)\n",
               N, best_BM, best_BN, best_BK, best_TM, best_TN, best_ms, best_tflops);
    }
    tuned = true;
}

MatmulVectorizedAuto::MatmulVectorizedAuto(int N, int blockDim)
    : N(N), blockDim(blockDim),
      best_BM(128), best_BN(128), best_BK(8), best_TM(8), best_TN(8),
      best_time_ms(0.0f), tuned(false) {}

void MatmulVectorizedAuto::execute(const float *d_A, const float *d_B, float *d_C) {
    if (!tuned) {
        tune(d_A, d_B, d_C);
    }
    launch(d_A, d_B, d_C, best_BM, best_BN, best_BK, best_TM, best_TN);
    cudaCheckError(cudaGetLastError());
}

MatmulVectorizedAuto::~MatmulVectorizedAuto() {}
