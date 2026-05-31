#ifndef MATMUL_VECTORIZED_H
#define MATMUL_VECTORIZED_H

#include "matmul_kernel.h"

// Vectorized Memory Access Optimization (Kernel 6 from siboehm.com)
//
// Key optimization: Use float4 (128-bit) loads from global and shared memory.
// Transpose As matrix during SMEM population for coalesced SMEM loads.
//
// VECTORIZATION:
// - float4 loads from GMEM (128-bit transactions)
// - Transpose As in SMEM: store as As[BK][BM] instead of As[BM][BK]
// - This enables coalesced SMEM reads during compute phase
//
// CONFIGURATION:
// - BM=128, BN=128, BK=8, TM=8, TN=8 (same as 2D blocktile)
// - Memory bandwidth is better utilized with 128-bit loads
//
// PERFORMANCE:
// Expected ~78.4% of cuBLAS (1.1x over 2D blocktile) due to vectorized loads.

class MatmulVectorized : public MatmulKernel {
private:
    int N;         // Matrix dimension (N×N matrices)
    int blockDim;  // Block dimension (unused, using fixed parameters)

public:
    MatmulVectorized(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulVectorized() override;
};

// Vectorized Memory Access with autotuning — sweeps (BM, BN, BK, TM, TN)
// candidates the first time execute() is called and caches the best.
//
// ~14 candidates explored across three axes:
//   - block tile size (BM, BN): 64..256, including asymmetric pairs
//   - K depth         (BK):    8, 16, or 32
//   - thread tile     (TM,TN): 4..16, asymmetric allowed
//
// Sweep policy: 2 warmup + 3 timed launches per candidate, median wins.
class MatmulVectorizedAuto : public MatmulKernel {
private:
    int N;
    int blockDim;
    int best_BM;
    int best_BN;
    int best_BK;
    int best_TM;
    int best_TN;
    float best_time_ms;
    bool tuned;

    void tune(const float *d_A, const float *d_B, float *d_C);
    void launch(const float *d_A, const float *d_B, float *d_C,
                int BM, int BN, int BK, int TM, int TN);

public:
    MatmulVectorizedAuto(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulVectorizedAuto() override;
};

#endif
