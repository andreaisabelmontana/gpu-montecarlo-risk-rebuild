/* cuda.cu — GPU backend (CUDA + cuRAND).
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ REQUIRES AN NVIDIA GPU AND THE CUDA TOOLKIT (nvcc).                        │
 * │ It will NOT build or run on machines without CUDA (e.g. AMD/Intel iGPUs). │
 * │ Build:  make cuda      (or)   nvcc -O3 -Iinclude src/cuda.cu src/common.c │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Design — one CUDA thread simulates many independent paths via a grid-stride
 * loop. Each thread owns a private cuRAND state (Philox4_32_10, a counter-based
 * RNG ideal for Monte Carlo: long period, no inter-thread correlation, no
 * shared state). The Cholesky factor is computed once on the host and lives in
 * __constant__ memory so every thread reads the correlation structure from the
 * fast constant cache. Per-path losses are written to a global array and the
 * host computes VaR/CVaR identically to the CPU backends, guaranteeing
 * cross-backend numerical consistency.
 *
 * Performance levers (the parts that delivered the headline speedup):
 *   - Philox cuRAND instead of XORWOW  -> fewer registers, higher occupancy.
 *   - Cholesky in __constant__ memory  -> broadcast reads, no global traffic.
 *   - Grid-stride loop                 -> launch a fixed grid, saturate the SMs.
 *   - Coalesced writes to losses[gid]  -> avoids the original RNG/reduction
 *                                         memory bottleneck.
 */
#include "montecarlo.h"
#include "timer.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* Model parameters in constant memory (read-only, broadcast to all threads). */
__constant__ int    d_n_assets;
__constant__ double d_T;
__constant__ double d_S0[MC_MAX_ASSETS];
__constant__ double d_mu[MC_MAX_ASSETS];
__constant__ double d_sigma[MC_MAX_ASSETS];
__constant__ double d_weights[MC_MAX_ASSETS];
__constant__ double d_chol[MC_MAX_ASSETS * MC_MAX_ASSETS];

__global__ void simulate_kernel(double *losses, size_t n_sims, unsigned long long seed) {
    size_t gid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    /* Per-thread Philox stream: same seed, distinct sequence number. */
    curandStatePhilox4_32_10_t st;
    curand_init(seed, gid, 0, &st);

    const int    N = d_n_assets;
    const double T = d_T;
    const double sqrtT = sqrt(T);

    for (size_t idx = gid; idx < n_sims; idx += stride) {
        double x[MC_MAX_ASSETS], z[MC_MAX_ASSETS];
        for (int i = 0; i < N; i++) x[i] = curand_normal_double(&st);

        double V_T = 0.0, V_0 = 0.0;
        for (int i = 0; i < N; i++) {
            double zi = 0.0;
            const double *Li = &d_chol[i * N];
            for (int j = 0; j <= i; j++) zi += Li[j] * x[j];
            z[i] = zi;
        }
        for (int i = 0; i < N; i++) {
            double drift = (d_mu[i] - 0.5 * d_sigma[i] * d_sigma[i]) * T;
            double ST = d_S0[i] * exp(drift + d_sigma[i] * sqrtT * z[i]);
            V_T += d_weights[i] * ST;
            V_0 += d_weights[i] * d_S0[i];
        }
        losses[idx] = V_0 - V_T;   /* coalesced write */
    }
}

int main(int argc, char **argv) {
    size_t n_sims = (argc > 1) ? strtoull(argv[1], NULL, 10) : 1000000ULL;
    unsigned long long seed = (argc > 2) ? strtoull(argv[2], NULL, 10) : 12345ULL;
    double alpha = (argc > 3) ? atof(argv[3]) : 0.95;

    Portfolio p;
    if (argc > 4) { if (portfolio_load(&p, argv[4]) != 0) { fprintf(stderr, "load failed\n"); return 1; } }
    else portfolio_demo(&p);

    /* Upload model to constant memory. */
    CUDA_CHECK(cudaMemcpyToSymbol(d_n_assets, &p.n_assets, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_T, &p.horizon_years, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_S0, p.S0, sizeof(double) * p.n_assets));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mu, p.mu, sizeof(double) * p.n_assets));
    CUDA_CHECK(cudaMemcpyToSymbol(d_sigma, p.sigma, sizeof(double) * p.n_assets));
    CUDA_CHECK(cudaMemcpyToSymbol(d_weights, p.weights, sizeof(double) * p.n_assets));
    CUDA_CHECK(cudaMemcpyToSymbol(d_chol, p.chol, sizeof(double) * p.n_assets * p.n_assets));

    double *d_losses, *h_losses;
    h_losses = (double *)malloc(n_sims * sizeof(double));
    CUDA_CHECK(cudaMalloc(&d_losses, n_sims * sizeof(double)));

    /* Launch a grid sized to the device, then grid-stride over all sims. */
    int block = 256;
    int grid;
    {
        int dev; cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        grid = prop.multiProcessorCount * 32;   /* oversubscribe the SMs */
    }

    double t0 = mc_wtime();
    simulate_kernel<<<grid, block>>>(d_losses, n_sims, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = mc_wtime();

    CUDA_CHECK(cudaMemcpy(h_losses, d_losses, n_sims * sizeof(double), cudaMemcpyDeviceToHost));

    McResult r = {0};
    r.seconds = t1 - t0;                       /* kernel time (excl. H2D copy) */
    r.sims_per_sec = (double)n_sims / r.seconds;
    r.V0 = portfolio_V0(&p);
    r.n_sims = n_sims;
    r.alpha = alpha;
    r.threads = grid * block;
    compute_var_cvar(h_losses, n_sims, alpha, &r.var, &r.cvar, &r.mean_loss);

    print_result_json("cuda", &r);

    cudaFree(d_losses);
    free(h_losses);
    return 0;
}
