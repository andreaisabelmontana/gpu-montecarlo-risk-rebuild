/* openmp.c — multi-core CPU backend via OpenMP.
 *
 * The simulation loop is embarrassingly parallel: each draw is independent.
 * The only correctness subtlety is the RNG — a single shared stream would
 * serialize and (worse) correlate threads. So each thread seeds its OWN
 * xoshiro256** stream from (seed + thread_id) and writes into a disjoint slice
 * of the loss array, giving deterministic, race-free, independent sub-streams.
 *
 * Usage: ./mc_openmp [n_sims] [seed] [alpha] [portfolio.csv]
 */
#include "montecarlo.h"
#include "timer.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    size_t n_sims = (argc > 1) ? strtoull(argv[1], NULL, 10) : 1000000ULL;
    uint64_t seed = (argc > 2) ? strtoull(argv[2], NULL, 10) : 12345ULL;
    double alpha  = (argc > 3) ? atof(argv[3]) : 0.95;

    Portfolio p;
    if (argc > 4) {
        if (portfolio_load(&p, argv[4]) != 0) {
            fprintf(stderr, "failed to load portfolio %s\n", argv[4]);
            return 1;
        }
    } else {
        portfolio_demo(&p);
    }

    double *losses = (double *)malloc(n_sims * sizeof(double));
    if (!losses) { fprintf(stderr, "OOM allocating %zu losses\n", n_sims); return 1; }

    int n_threads = 1;
    double t0 = mc_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        n_threads = omp_get_num_threads();

        /* Per-thread independent RNG stream. */
        normal_t rng;
        normal_init(&rng, seed + (uint64_t)tid * 0x9E3779B97F4A7C15ULL);
        double z[MC_MAX_ASSETS];

        /* Static schedule -> each thread owns a contiguous, disjoint slice,
         * so the mapping (index -> thread -> RNG stream) is reproducible. */
        #pragma omp for schedule(static)
        for (long long i = 0; i < (long long)n_sims; i++)
            losses[i] = mc_simulate_loss(&p, &rng, z);
    }

    double t1 = mc_wtime();

    McResult r = {0};
    r.seconds = t1 - t0;
    r.sims_per_sec = (double)n_sims / r.seconds;
    r.V0 = portfolio_V0(&p);
    r.n_sims = n_sims;
    r.alpha = alpha;
    r.threads = n_threads;
    compute_var_cvar(losses, n_sims, alpha, &r.var, &r.cvar, &r.mean_loss);

    print_result_json("openmp", &r);
    free(losses);
    return 0;
}
