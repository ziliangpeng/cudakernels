#ifndef MATMUL_2D_BLOCKTILE_H
#define MATMUL_2D_BLOCKTILE_H

#include "matmul_kernel.h"

// 2D Block Tiling Optimization (Kernel 5 from siboehm.com)
//
// Key optimization: Each thread computes a TM x TN tile of output elements,
// using the outer product formulation in the innermost loop.
//
// CONFIGURATION:
// - BM=128, BN=128, BK=8, TM=8, TN=8
// - Each thread computes 8x8 = 64 C elements
// - Threads per block: (BM/TM) * (BN/TN) = 16 * 16 = 256
//
// OUTER PRODUCT:
// - Load TM elements of A column and TN elements of B row into registers
// - Compute outer product: regC[i][j] += regA[i] * regB[j]
// - This maximizes register reuse
//
// PERFORMANCE:
// Expected ~68.7% of cuBLAS (1.9x over 1D blocktile) due to 2D tiling.

class Matmul2DBlocktile : public MatmulKernel {
private:
    int N;         // Matrix dimension (N×N matrices)
    int blockDim;  // Block dimension (unused, using fixed BM/BN/BK/TM/TN)

public:
    Matmul2DBlocktile(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~Matmul2DBlocktile() override;
};

// 2D Block Tiling with autotuning — sweeps (BM, BN, BK, TM, TN) candidates the
// first time execute() is called and caches the best for subsequent launches.
//
// 19 candidates explored across three axes:
//   - block tile size (BM, BN): 64..256, including asymmetric pairs
//   - K depth         (BK):    8 or 16
//   - thread tile     (TM,TN): 4..16, asymmetric allowed
//
// Sweep policy: 2 warmup + 3 timed launches per candidate, median wins.
// Warmup is followed by cudaGetLastError to skip failing configs cleanly.
class Matmul2DBlocktileAuto : public MatmulKernel {
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
    Matmul2DBlocktileAuto(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~Matmul2DBlocktileAuto() override;
};

#endif
