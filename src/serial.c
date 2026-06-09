/* serial.c — single-threaded reference backend.
 *
 * The numerical ground truth. Every other backend must reproduce its VaR/CVaR
 * to within Monte Carlo noise. Usage:
 *     ./mc_serial [n_sims] [seed] [alpha] [portfolio.csv]
 */
#include "montecarlo.h"
#include "timer.h"

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

    normal_t rng;
    normal_init(&rng, seed);
    double z[MC_MAX_ASSETS];

    double t0 = mc_wtime();
    for (size_t i = 0; i < n_sims; i++)
        losses[i] = mc_simulate_loss(&p, &rng, z);
    double t1 = mc_wtime();

    McResult r = {0};
    r.seconds = t1 - t0;
    r.sims_per_sec = (double)n_sims / r.seconds;
    r.V0 = portfolio_V0(&p);
    r.n_sims = n_sims;
    r.alpha = alpha;
    r.threads = 1;
    compute_var_cvar(losses, n_sims, alpha, &r.var, &r.cvar, &r.mean_loss);

    print_result_json("serial", &r);
    free(losses);
    return 0;
}
