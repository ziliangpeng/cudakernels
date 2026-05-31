#ifndef MATMUL_WARPTILE_H
#define MATMUL_WARPTILE_H

#include "matmul_kernel.h"

// Warp Tiling Optimization (Kernel 10 from siboehm.com)
//
// Key optimization: Add warp-level tiling between block and thread levels.
// This organizes computation to maximize warp scheduler utilization.
//
// THREE-LEVEL HIERARCHY:
// - Block tile: BM x BN (e.g., 128 x 128)
// - Warp tile: WM x WN (e.g., 64 x 64)
// - Thread tile: TM x TN (e.g., 8 x 8)
//
// WARP ORGANIZATION:
// - Each warp (32 threads) computes a WM x WN tile
// - Warps are arranged to maximize scheduler efficiency
// - 4 warps per SM, each handling different output regions
//
// PERFORMANCE:
// Expected ~93.7% of cuBLAS (1.1x over vectorized) - near peak performance.

class MatmulWarptile : public MatmulKernel {
private:
    int N;         // Matrix dimension (N×N matrices)
    int blockDim;  // Block dimension (unused, using fixed parameters)

public:
    MatmulWarptile(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWarptile() override;
};

// Warp Tiling with autotuning — sweeps (BM, BN, BK, TM, TN, WM, WN) candidates
// the first time execute() is called and caches the best for subsequent launches.
//
// ~15 candidates explored across these axes:
//   - block tile size  (BM, BN):  128..256, tall/wide included
//   - K depth           (BK):     8 or 16
//   - thread tile       (TM, TN): 2..16
//   - warp tile         (WM, WN): 32..128
//
// Within-warp layout is fixed: WARP_THREAD_M=4, WARP_THREAD_N=8, so each warp
// assigns 4 threads along M and 8 threads along N.  When WM/(4*TM) > 1 or
// WN/(8*TN) > 1, each thread iterates over multiple TM×TN subtiles.
//
// Sweep policy: 2 warmup + 3 timed launches per candidate, median wins.
class MatmulWarptileAuto : public MatmulKernel {
private:
    int N;
    int blockDim;
    int best_BM, best_BN, best_BK, best_TM, best_TN, best_WM, best_WN;
    float best_time_ms;
    bool tuned;

    void tune(const float *d_A, const float *d_B, float *d_C);
    void launch(const float *d_A, const float *d_B, float *d_C,
                int BM, int BN, int BK, int TM, int TN, int WM, int WN);

public:
    MatmulWarptileAuto(int N, int blockDim);
    void execute(const float *d_A, const float *d_B, float *d_C) override;
    ~MatmulWarptileAuto() override;
};

#endif
