#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <algorithm>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "matmul_naive.h"
#include "matmul_coalesced.h"
#include "matmul_smem.h"
#include "matmul_1d_blocktile.h"
#include "matmul_2d_blocktile.h"
#include "matmul_vectorized.h"
#include "matmul_warptile.h"
#include "matmul_cublas.h"
#include "matmul_wmma.h"
#include "matmul_wmma_bf16.h"
#include "matmul_cublas_bf16.h"
#include "matrix_init.h"

// ============================================================================
// BENCHMARK MODE: Data Structures and Constants
// ============================================================================

// Benchmark configuration
const char* BENCHMARK_METHODS[] = {
    "naive",
    "coalesced",
    "smem",
    "1d_blocktile",
    "1d_blocktile_auto",
    "2d_blocktile",
    "2d_blocktile_auto",
    "vectorized",
    "warptile",
    "cublas",
    "wmma",
    "wmma_bf16",
    "cublas_bf16"
};
const int NUM_METHODS = 13;

const int BENCHMARK_SIZES[] = {64, 128, 256, 512, 1024, 2048};
const int NUM_SIZES = sizeof(BENCHMARK_SIZES) / sizeof(BENCHMARK_SIZES[0]);
const char* SIZE_LABELS[] = {"64", "128", "256", "512", "1K", "2K"};

// Result structures
struct BenchmarkResult {
    float median_time_ms;    // -1.0 if skipped/failed
    double gflops;           // Computed GFLOPS
    double mfu_percent;      // Model FLOPS Utilization (%)
    bool skipped;
    std::string skip_reason;  // Modern C++ string (no buffer overflow possible)
};

struct VerificationResult {
    bool passed;
    double max_rel_error;    // -1.0 if skipped
    bool has_nan_inf;
    bool skipped;
};

// ============================================================================
// GPU Performance Detection
// ============================================================================

// Returns theoretical peak GFLOPS including Tensor Cores for modern GPUs
double get_gpu_peak_gflops() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int numSMs = deviceProp.multiProcessorCount;
    float clockRate_GHz = deviceProp.clockRate / 1e6f;  // kHz to GHz
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    double peak_gflops = 0.0;
    const char* compute_type = "FP32";

    // Use Tensor Core peaks for GPUs that have them (SM70+)
    // These represent TF32/FP16 Tensor Core throughput which libraries like cuBLAS use

    if (major == 9 && minor == 0) {
        // Hopper (H100)
        if (strstr(deviceProp.name, "H100") != nullptr) {
            peak_gflops = 495000.0;  // TF32 Tensor Core peak
            compute_type = "TF32 Tensor Core";
        } else {
            peak_gflops = numSMs * 2.0 * clockRate_GHz * 2.0;
        }
    } else if (major == 8 && minor == 9) {
        // Ada Lovelace (RTX 40 series)
        if (strstr(deviceProp.name, "4090") != nullptr) {
            peak_gflops = 82580.0;  // FP32 peak (no sparse)
            compute_type = "FP32";
        } else if (strstr(deviceProp.name, "4080") != nullptr) {
            peak_gflops = 48740.0;
            compute_type = "FP32";
        } else {
            peak_gflops = numSMs * 2.0 * clockRate_GHz * 2.0;
        }
    } else if (major == 8 && minor == 6) {
        // Ampere (RTX 30 series)
        if (strstr(deviceProp.name, "3090") != nullptr) {
            peak_gflops = 35580.0;  // FP32 peak
            compute_type = "FP32";
        } else if (strstr(deviceProp.name, "3080") != nullptr) {
            peak_gflops = 29770.0;
            compute_type = "FP32";
        } else {
            peak_gflops = numSMs * 2.0 * clockRate_GHz * 2.0;
        }
    } else if (major == 8 && minor == 0) {
        // Ampere (A100)
        if (strstr(deviceProp.name, "A100") != nullptr) {
            peak_gflops = 156000.0;  // TF32 Tensor Core peak
            compute_type = "TF32 Tensor Core";
        } else {
            peak_gflops = numSMs * 2.0 * clockRate_GHz * 2.0;
        }
    } else if (major == 7 && minor == 5) {
        // Turing (RTX 20 series, T4)
        if (strstr(deviceProp.name, "T4") != nullptr) {
            peak_gflops = 8100.0;  // FP32 peak
            compute_type = "FP32";
        } else {
            peak_gflops = numSMs * 2.0 * clockRate_GHz * 2.0;
        }
    } else if (major == 7 && minor == 0) {
        // Volta (V100)
        if (strstr(deviceProp.name, "V100") != nullptr) {
            peak_gflops = 15700.0;  // FP32 peak
            compute_type = "FP32";
        } else {
            peak_gflops = numSMs * 2.0 * clockRate_GHz * 2.0;
        }
    } else {
        // Fallback: calculate from hardware specs (older GPUs or unknown)
        peak_gflops = numSMs * 2.0 * clockRate_GHz * 2.0;
        compute_type = "FP32 (calculated)";
    }

    // Print GPU info for user reference
    printf("GPU: %s (SM%d%d, %d SMs @ %.2f GHz)\n",
           deviceProp.name, major, minor, numSMs, clockRate_GHz);
    printf("Theoretical Peak: %.1f GFLOPS (%s)\n", peak_gflops, compute_type);
    printf("=============================================================================\n\n");

    return peak_gflops;
}

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -n, --size N              Set matrix dimension N×N (default: 512)\n");
    printf("  -b, --block-dim N         Set block dimension N×N (default: 16)\n");
    printf("  -m, --method METHOD       Matmul method: 'naive' (default: naive)\n");
    printf("  -v, --verify              Enable result verification\n");
    printf("  -h, --help                Show this help message\n");
    printf("\nMethods (ordered by optimization level):\n");
    printf("  naive:         Naive triple-nested loop (simple, unoptimized)\n");
    printf("  coalesced:     Global memory coalescing optimization\n");
    printf("  smem:          Shared memory tiling\n");
    printf("  1d_blocktile:  1D block tiling (TM=8 elements per thread)\n");
    printf("  2d_blocktile:  2D block tiling (TM=TN=8, 64 elements per thread)\n");
    printf("  vectorized:    float4 vectorized memory access\n");
    printf("  warptile:      Warp-level tiling (near-optimal)\n");
    printf("  cublas:        NVIDIA cuBLAS library (highly optimized)\n");
    printf("  wmma:          WMMA Tensor Core FP16 (Volta+ GPUs)\n");
    printf("  wmma_bf16:     WMMA Tensor Core BF16 (Ampere+ GPUs)\n");
    printf("  cublas_bf16:   cuBLAS BF16 Tensor Core (Ampere+ GPUs)\n");
    printf("\nSpecial method:\n");
    printf("  all:           Run comprehensive benchmark across all methods and sizes\n");
    printf("                 Tests sizes: 64, 128, 256, 512, 1K, 2K\n");
    printf("                 Iterations: 100 per test\n");
    printf("                 Output: Formatted performance table\n");
    printf("\n                 When combined with --verify:\n");
    printf("                   - Validates correctness against CPU reference\n");
    printf("                   - Prints accuracy comparison table\n");
    printf("\nExample usage:\n");
    printf("  %s --method all                # Performance benchmark only\n", program_name);
    printf("  %s --method all --verify       # Performance + verification\n", program_name);
    printf("  %s --method naive -n 256 --verify  # Test single method\n", program_name);
}

// ============================================================================
// BENCHMARK MODE: Helper Functions
// ============================================================================

// Helper: Get batched time for a method (returns -1.0 on failure)
// This measures throughput by queuing all kernels and timing the batch,
// which removes per-iteration event overhead and shows pure kernel performance
float get_median_time(MatmulKernel *kernel, const float *d_A, const float *d_B,
                      float *d_C, int num_iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup: run a few iterations to ensure kernel is compiled and caches are warm
    for (int i = 0; i < 10; i++) {
        kernel->execute(d_A, d_B, d_C);
    }
    cudaDeviceSynchronize();

    // Batched timing: queue all iterations, then sync once at the end
    // This removes per-iteration event recording overhead
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        kernel->execute(d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_time_ms;
    cudaEventElapsedTime(&total_time_ms, start, stop);

    float avg_time_ms = total_time_ms / num_iterations;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return avg_time_ms;
}

// Helper: Run verification for a method
VerificationResult verify_method(MatmulKernel *kernel, const float *d_A,
                                 const float *d_B, float *d_C,
                                 const float *h_A, const float *h_B,
                                 float *h_C, const float *h_expected, int N) {
    VerificationResult result;
    result.skipped = false;
    result.passed = false;
    result.max_rel_error = -1.0;
    result.has_nan_inf = false;

    // Run kernel once
    kernel->execute(d_A, d_B, d_C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.skipped = true;
        return result;
    }

    // Transfer output
    err = cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        result.skipped = true;
        return result;
    }

    // Check for NaN/Inf
    for (int i = 0; i < N * N; i++) {
        if (isnan(h_C[i]) || isinf(h_C[i])) {
            result.has_nan_inf = true;
            result.passed = false;
            return result;
        }
    }

    // Compare against CPU reference
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;

    for (int i = 0; i < N * N; i++) {
        double abs_error = fabs(h_C[i] - h_expected[i]);
        double rel_error = abs_error / (fabs(h_expected[i]) + 1e-10);

        if (abs_error > max_abs_error) max_abs_error = abs_error;
        if (rel_error > max_rel_error) max_rel_error = rel_error;
    }

    result.max_rel_error = max_rel_error;
    result.passed = (max_rel_error < 1e-4);

    return result;
}

// ============================================================================
// BENCHMARK MODE: Table Printing Functions
// ============================================================================

void print_performance_table(BenchmarkResult results[][NUM_SIZES]) {
    printf("\n\n");
    printf("=============================================================================\n");
    printf("                    MATMUL PERFORMANCE BENCHMARK\n");
    printf("=============================================================================\n");
    printf("Iterations per test: 100\n");
    printf("Metrics: Time (ms) | GFLOPS | MFU (%%)\n");
    printf("Method: Batched timing (100 kernels queued, single sync)\n\n");

    // Print header
    printf("%-15s |", "Method");
    for (int s = 0; s < NUM_SIZES; s++) {
        printf(" %-9s|", SIZE_LABELS[s]);
    }
    printf("\n");

    // Print separator
    printf("----------------|");
    for (int s = 0; s < NUM_SIZES; s++) {
        printf("----------|");
    }
    printf("\n");

    // Print data rows - 3 rows per method
    for (int m = 0; m < NUM_METHODS; m++) {
        // Row 1: Method name and time (ms)
        char label[32];
        snprintf(label, sizeof(label), "%s (time)", BENCHMARK_METHODS[m]);
        printf("%-15s |", label);
        for (int s = 0; s < NUM_SIZES; s++) {
            if (results[m][s].skipped) {
                printf(" %9s|", "SKIPPED");
            } else if (results[m][s].median_time_ms < 0.0f) {
                printf(" %9s|", "FAILED");
            } else {
                printf(" %9.3f|", results[m][s].median_time_ms);
            }
        }
        printf("\n");

        // Row 2: GFLOPS (with T/G suffix for readability)
        printf("%7s(gflops) |", "");
        for (int s = 0; s < NUM_SIZES; s++) {
            if (results[m][s].skipped || results[m][s].median_time_ms < 0.0f) {
                printf(" %9s|", "-");
            } else {
                double gflops = results[m][s].gflops;
                if (gflops >= 1000.0) {
                    printf(" (%6.1f T)|", gflops / 1000.0);
                } else {
                    printf(" (%6.1f G)|", gflops);
                }
            }
        }
        printf("\n");

        // Row 3: MFU %
        printf("%10s(mfu) |", "");
        for (int s = 0; s < NUM_SIZES; s++) {
            if (results[m][s].skipped || results[m][s].median_time_ms < 0.0f) {
                printf(" %9s|", "-");
            } else {
                printf(" (%6.1f%%)|", results[m][s].mfu_percent);
            }
        }
        printf("\n");
    }
    printf("=============================================================================\n");
}

void print_verification_table(VerificationResult results[][NUM_SIZES]) {
    printf("\n");
    printf("=============================================================================\n");
    printf("                    MATMUL VERIFICATION RESULTS\n");
    printf("=============================================================================\n");
    printf("Reference: CPU matmul (double precision)\n");
    printf("Threshold: Max relative error < 1e-4\n\n");

    // Print header
    printf("%-15s", "Method");
    for (int s = 0; s < NUM_SIZES; s++) {
        printf(" | %-13s", SIZE_LABELS[s]);
    }
    printf(" |\n");

    // Print separator
    printf("%-15s", "---------------");
    for (int s = 0; s < NUM_SIZES; s++) {
        printf("-|--------------");
    }
    printf("-|\n");

    // Print data rows
    for (int m = 0; m < NUM_METHODS; m++) {
        printf("%-15s", BENCHMARK_METHODS[m]);
        for (int s = 0; s < NUM_SIZES; s++) {
            if (results[m][s].skipped) {
                printf(" | %-13s", "SKIPPED");
            } else if (results[m][s].has_nan_inf) {
                printf(" | %-13s", "FAIL (NaN)");
            } else if (results[m][s].passed) {
                printf(" | PASS (%.2e)", results[m][s].max_rel_error);
            } else {
                printf(" | FAIL (%.2e)", results[m][s].max_rel_error);
            }
        }
        printf(" |\n");
    }

    printf("\nLegend:\n");
    printf("- PASS (error): Verification passed, shows max relative error\n");
    printf("- FAIL (NaN):   Output contains NaN or Inf\n");
    printf("- FAIL (error): Error exceeds threshold\n");
    printf("- SKIPPED:      Method failed to execute\n");
    printf("=============================================================================\n");
}

// Main benchmark function: test all methods across all sizes
void benchmark_all_methods(int blockDim, bool verify) {
    printf("\n=============================================================================\n");
    printf("                    RUNNING COMPREHENSIVE BENCHMARK\n");
    printf("=============================================================================\n");

    // Detect GPU and calculate peak performance
    double peak_gflops = get_gpu_peak_gflops();

    printf("Methods to test: %d\n", NUM_METHODS);
    printf("Sizes to test: %d (64, 128, 256, 512, 1K)\n", NUM_SIZES);
    printf("Iterations per test: 100\n");
    printf("Block dimension: %d×%d (%d threads)\n", blockDim, blockDim, blockDim * blockDim);
    if (verify) {
        printf("Verification: ENABLED\n");
    }
    printf("=============================================================================\n\n");

    // Allocate result arrays
    BenchmarkResult perf_results[NUM_METHODS][NUM_SIZES];
    VerificationResult verify_results[NUM_METHODS][NUM_SIZES];

    // Initialize all results as skipped by default
    for (int m = 0; m < NUM_METHODS; m++) {
        for (int s = 0; s < NUM_SIZES; s++) {
            perf_results[m][s].median_time_ms = -1.0f;
            perf_results[m][s].gflops = 0.0;
            perf_results[m][s].mfu_percent = 0.0;
            perf_results[m][s].skipped = true;
            perf_results[m][s].skip_reason = "Not run";

            verify_results[m][s].skipped = true;
            verify_results[m][s].passed = false;
            verify_results[m][s].max_rel_error = -1.0;
            verify_results[m][s].has_nan_inf = false;
        }
    }

    // Run benchmarks for each method and size
    for (int m = 0; m < NUM_METHODS; m++) {
        const char *method = BENCHMARK_METHODS[m];
        printf("Testing method: %s\n", method);

        for (int s = 0; s < NUM_SIZES; s++) {
            int N = BENCHMARK_SIZES[s];

            // Skip large sizes for naive kernel (too slow)
            if (strcmp(method, "naive") == 0 && N >= 1024) {
                printf("  Size: %s (%d×%d)... SKIPPED (too slow for naive)\n",
                       SIZE_LABELS[s], N, N);
                perf_results[m][s].skipped = true;
                perf_results[m][s].median_time_ms = -1.0f;
                perf_results[m][s].gflops = 0.0;
                perf_results[m][s].mfu_percent = 0.0;
                perf_results[m][s].skip_reason = "Size too large for naive kernel";
                verify_results[m][s].skipped = true;
                continue;
            }

            printf("  Size: %s (%d×%d)... ", SIZE_LABELS[s], N, N);
            fflush(stdout);

            // Allocate host memory
            float *h_A, *h_B;
            allocateAndInitMatrices(&h_A, &h_B, N);
            float *h_C = (float*)malloc(N * N * sizeof(float));
            float *h_expected = nullptr;

            if (!h_A || !h_B || !h_C) {
                printf("SKIPPED (host allocation failed)\n");
                perf_results[m][s].skip_reason = "Host allocation failed";
                if (h_A) free(h_A);
                if (h_B) free(h_B);
                if (h_C) free(h_C);
                continue;
            }

            // Allocate device memory
            float *d_A, *d_B, *d_C;
            cudaError_t err = cudaMalloc(&d_A, N * N * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                perf_results[m][s].skip_reason = "Device allocation failed";
                free(h_A);
                free(h_B);
                free(h_C);
                continue;
            }

            err = cudaMalloc(&d_B, N * N * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                perf_results[m][s].skip_reason = "Device allocation failed";
                free(h_A);
                free(h_B);
                free(h_C);
                cudaFree(d_A);
                continue;
            }

            err = cudaMalloc(&d_C, N * N * sizeof(float));
            if (err != cudaSuccess) {
                printf("SKIPPED (device allocation failed)\n");
                perf_results[m][s].skip_reason = "Device allocation failed";
                free(h_A);
                free(h_B);
                free(h_C);
                cudaFree(d_A);
                cudaFree(d_B);
                continue;
            }

            // Transfer to device
            cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

            // Compute CPU reference if verification enabled
            if (verify) {
                h_expected = (float*)malloc(N * N * sizeof(float));
                if (h_expected) {
                    matmul_cpu_reference(h_A, h_B, h_expected, N);
                }
            }

            // Try to instantiate kernel
            MatmulKernel *kernel = nullptr;
            try {
                if (strcmp(method, "naive") == 0) {
                    kernel = new MatmulNaive(N, blockDim);
                } else if (strcmp(method, "coalesced") == 0) {
                    kernel = new MatmulCoalesced(N, blockDim);
                } else if (strcmp(method, "smem") == 0) {
                    kernel = new MatmulSmem(N, blockDim);
                } else if (strcmp(method, "1d_blocktile") == 0) {
                    kernel = new Matmul1DBlocktile(N, blockDim);
                } else if (strcmp(method, "1d_blocktile_auto") == 0) {
                    kernel = new Matmul1DBlocktileAuto(N, blockDim);
                } else if (strcmp(method, "2d_blocktile") == 0) {
                    kernel = new Matmul2DBlocktile(N, blockDim);
                } else if (strcmp(method, "2d_blocktile_auto") == 0) {
                    kernel = new Matmul2DBlocktileAuto(N, blockDim);
                } else if (strcmp(method, "vectorized") == 0) {
                    kernel = new MatmulVectorized(N, blockDim);
                } else if (strcmp(method, "warptile") == 0) {
                    kernel = new MatmulWarptile(N, blockDim);
                } else if (strcmp(method, "cublas") == 0) {
                    kernel = new MatmulCublas(N, blockDim);
                } else if (strcmp(method, "wmma") == 0) {
                    kernel = new MatmulWMMA(N, blockDim);
                } else if (strcmp(method, "wmma_bf16") == 0) {
                    kernel = new MatmulWmmaBf16(N, blockDim);
                } else if (strcmp(method, "cublas_bf16") == 0) {
                    kernel = new MatmulCublasBf16(N, blockDim);
                }

                if (!kernel) {
                    printf("SKIPPED (unknown method)\n");
                    perf_results[m][s].skip_reason = "Unknown method";
                } else {
                    // Run performance benchmark
                    float median_time = get_median_time(kernel, d_A, d_B, d_C, 100);

                    if (median_time < 0.0f) {
                        printf("SKIPPED (benchmark failed)\n");
                        perf_results[m][s].skip_reason = "Benchmark failed";
                    } else {
                        perf_results[m][s].median_time_ms = median_time;
                        perf_results[m][s].skipped = false;
                        perf_results[m][s].skip_reason = "";

                        // Calculate GFLOPS and MFU
                        int N = BENCHMARK_SIZES[s];
                        double flops = 2.0 * N * N * N - N * N;  // Matrix multiply FLOPs
                        perf_results[m][s].gflops = flops / (median_time * 1e6);
                        perf_results[m][s].mfu_percent = (perf_results[m][s].gflops / peak_gflops) * 100.0;

                        // Run verification if enabled
                        if (verify && h_expected) {
                            VerificationResult vr = verify_method(kernel, d_A, d_B, d_C,
                                                                  h_A, h_B, h_C, h_expected, N);
                            verify_results[m][s] = vr;

                            if (vr.has_nan_inf) {
                                printf("%.3f ms [FAIL: NaN/Inf]\n", median_time);
                            } else if (vr.passed) {
                                printf("%.3f ms [PASS: %.2e]\n", median_time, vr.max_rel_error);
                            } else if (!vr.skipped) {
                                printf("%.3f ms [FAIL: %.2e]\n", median_time, vr.max_rel_error);
                            } else {
                                printf("%.3f ms [verify skipped]\n", median_time);
                            }
                        } else {
                            printf("%.3f ms\n", median_time);
                        }
                    }

                    delete kernel;
                }
            } catch (const std::exception &e) {
                printf("SKIPPED (exception: %s)\n", e.what());
                perf_results[m][s].skip_reason = std::string("Exception: ") + e.what();
                if (kernel) delete kernel;
            } catch (...) {
                printf("SKIPPED (unknown exception)\n");
                perf_results[m][s].skip_reason = "Unknown exception";
                if (kernel) delete kernel;
            }

            // Cleanup
            free(h_A);
            free(h_B);
            free(h_C);
            if (h_expected) free(h_expected);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
        }
        printf("\n");
    }

    // Print results
    printf("\n");
    print_performance_table(perf_results);

    if (verify) {
        printf("\n");
        print_verification_table(verify_results);
    }
}

// Matrix multiplication operation (single method mode)
void matmul_op(int N, int blockDim, bool verify, const char *method) {
    // Detect GPU and get peak performance (for MFU calculation)
    double peak_gflops = get_gpu_peak_gflops();

    printf("Matrix multiplication of %d×%d matrices\n", N, N);
    printf("Method: %s\n", method);
    printf("Block dimension: %d×%d (%d threads)\n", blockDim, blockDim, blockDim * blockDim);

    if (verify) {
        printf("Verification enabled\n");
    }

    // Allocate and initialize input matrices
    float *h_A, *h_B;
    allocateAndInitMatrices(&h_A, &h_B, N);

    // Allocate output matrix on host
    float *h_C = (float*)malloc(N * N * sizeof(float));
    if (!h_C) {
        fprintf(stderr, "Failed to allocate host output memory\n");
        exit(EXIT_FAILURE);
    }

    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc(&d_A, N * N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_B, N * N * sizeof(float)));
    cudaCheckError(cudaMalloc(&d_C, N * N * sizeof(float)));

    // Transfer to device
    cudaCheckError(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // Configure timing
    const int num_iterations = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running matmul %d times to collect statistics...\n", num_iterations);

    // Instantiate kernel
    MatmulKernel *kernel = nullptr;

    if (strcmp(method, "naive") == 0) {
        kernel = new MatmulNaive(N, blockDim);
    } else if (strcmp(method, "coalesced") == 0) {
        kernel = new MatmulCoalesced(N, blockDim);
    } else if (strcmp(method, "smem") == 0) {
        kernel = new MatmulSmem(N, blockDim);
    } else if (strcmp(method, "1d_blocktile") == 0) {
        kernel = new Matmul1DBlocktile(N, blockDim);
    } else if (strcmp(method, "1d_blocktile_auto") == 0) {
        kernel = new Matmul1DBlocktileAuto(N, blockDim);
    } else if (strcmp(method, "2d_blocktile") == 0) {
        kernel = new Matmul2DBlocktile(N, blockDim);
    } else if (strcmp(method, "2d_blocktile_auto") == 0) {
        kernel = new Matmul2DBlocktileAuto(N, blockDim);
    } else if (strcmp(method, "vectorized") == 0) {
        kernel = new MatmulVectorized(N, blockDim);
    } else if (strcmp(method, "warptile") == 0) {
        kernel = new MatmulWarptile(N, blockDim);
    } else if (strcmp(method, "cublas") == 0) {
        kernel = new MatmulCublas(N, blockDim);
    } else if (strcmp(method, "wmma") == 0) {
        kernel = new MatmulWMMA(N, blockDim);
    } else if (strcmp(method, "wmma_bf16") == 0) {
        kernel = new MatmulWmmaBf16(N, blockDim);
    } else if (strcmp(method, "cublas_bf16") == 0) {
        kernel = new MatmulCublasBf16(N, blockDim);
    }

    if (kernel) {
        // Class-based API: Time ONLY kernel execution (no setup/teardown overhead)
        // Warmup
        for (int i = 0; i < 10; i++) {
            kernel->execute(d_A, d_B, d_C);
        }
        cudaDeviceSynchronize();

        // Timing loop (batched)
        cudaEventRecord(start);
        for (int i = 0; i < num_iterations; i++) {
            kernel->execute(d_A, d_B, d_C);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_time_ms;
        cudaEventElapsedTime(&total_time_ms, start, stop);
        float avg_time_ms = total_time_ms / num_iterations;

        printf("Average time: %.3f ms\n", avg_time_ms);

        // Calculate GFLOPS
        // For N×N matmul: (2*N³ - N²) FLOPs
        double flops = 2.0 * N * N * N - N * N;
        double gflops = (flops / (avg_time_ms * 1e6));
        printf("Performance: %.2f GFLOPS\n", gflops);

        // Calculate and display MFU
        double mfu_percent = (gflops / peak_gflops) * 100.0;
        printf("MFU: %.1f%% of theoretical peak\n", mfu_percent);

        delete kernel;
    } else {
        fprintf(stderr, "Unknown method: %s\n", method);
        exit(EXIT_FAILURE);
    }

    cudaCheckError(cudaGetLastError());

    // Transfer result back for verification
    if (verify) {
        cudaCheckError(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

        printf("\nVerifying results...\n");

        // Calculate expected result on CPU
        float *h_expected = (float*)malloc(N * N * sizeof(float));
        if (!h_expected) {
            fprintf(stderr, "Failed to allocate CPU reference memory\n");
            exit(EXIT_FAILURE);
        }
        matmul_cpu_reference(h_A, h_B, h_expected, N);

        // Check for NaN/Inf
        bool has_nan_inf = false;
        for (int i = 0; i < N * N; i++) {
            if (isnan(h_C[i]) || isinf(h_C[i])) {
                has_nan_inf = true;
                break;
            }
        }

        if (has_nan_inf) {
            printf("FAILED: Output contains NaN or Inf\n");
        } else {
            // Compare against CPU reference
            double max_abs_error = 0.0;
            double max_rel_error = 0.0;

            for (int i = 0; i < N * N; i++) {
                double abs_error = fabs(h_C[i] - h_expected[i]);
                double rel_error = abs_error / (fabs(h_expected[i]) + 1e-10);

                if (abs_error > max_abs_error) max_abs_error = abs_error;
                if (rel_error > max_rel_error) max_rel_error = rel_error;
            }

            printf("Max absolute error: %.2e\n", max_abs_error);
            printf("Max relative error: %.2e\n", max_rel_error);

            if (max_rel_error < 1e-4) {
                printf("PASSED: Results match CPU reference\n");
            } else {
                printf("FAILED: Error exceeds threshold (1e-4)\n");
            }
        }

        free(h_expected);
    }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char *argv[]) {
    // Default parameters
    int N = 512;
    int blockDim = 16;
    const char *method = "naive";
    bool verify = false;

    // Parse command line arguments
    static struct option long_options[] = {
        {"size",      required_argument, 0, 'n'},
        {"block-dim", required_argument, 0, 'b'},
        {"method",    required_argument, 0, 'm'},
        {"verify",    no_argument,       0, 'v'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "n:b:m:vh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'n':
                N = atoi(optarg);
                break;
            case 'b':
                blockDim = atoi(optarg);
                break;
            case 'm':
                method = optarg;
                break;
            case 'v':
                verify = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // Check if "all" method is requested
    if (strcmp(method, "all") == 0) {
        benchmark_all_methods(blockDim, verify);
    } else {
        matmul_op(N, blockDim, verify, method);
    }

    return 0;
}
