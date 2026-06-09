/* montecarlo.h — shared model for the multi-backend Monte Carlo risk engine.
 *
 * The engine prices a portfolio of correlated assets under geometric Brownian
 * motion and estimates Value-at-Risk (VaR) and Conditional VaR / Expected
 * Shortfall (CVaR) over a horizon. Correlated shocks are produced from i.i.d.
 * standard normals via a Cholesky factor of the correlation matrix.
 *
 * The hot path (RNG + single-path simulation) lives here as `static inline`
 * so every CPU backend compiles it directly and the optimizer can go to town.
 * The Serial, OpenMP and CUDA backends differ ONLY in how they parallelize the
 * loop over independent simulations — the math below is identical, which is
 * what guarantees numerical consistency across backends.
 */
#ifndef MONTECARLO_H
#define MONTECARLO_H

#include <math.h>
#include <stdint.h>
#include <stddef.h>

#define MC_MAX_ASSETS 32

/* ------------------------------------------------------------------ Portfolio */
typedef struct {
    int    n_assets;
    double S0[MC_MAX_ASSETS];       /* spot prices                    */
    double mu[MC_MAX_ASSETS];       /* annualized drift               */
    double sigma[MC_MAX_ASSETS];    /* annualized volatility          */
    double weights[MC_MAX_ASSETS];  /* portfolio units per asset      */
    double corr[MC_MAX_ASSETS * MC_MAX_ASSETS];  /* correlation matrix */
    double chol[MC_MAX_ASSETS * MC_MAX_ASSETS];  /* lower Cholesky L   */
    double horizon_years;
} Portfolio;

typedef struct {
    double seconds;
    double sims_per_sec;
    double V0;
    double mean_loss;
    double var;        /* VaR at `alpha`                 */
    double cvar;       /* CVaR / Expected Shortfall      */
    double alpha;
    size_t n_sims;
    int    threads;
} McResult;

/* ------------------------------------------------------------------- RNG -----
 * xoshiro256** seeded by splitmix64. Each parallel stream gets its own state
 * seeded from a distinct value, so threads draw independent sub-sequences.
 */
typedef struct { uint64_t s[4]; } rng_t;

static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline void rng_seed(rng_t *r, uint64_t seed) {
    uint64_t sm = seed;
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&sm);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s;
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

/* Uniform in (0,1) — 53-bit mantissa, never exactly 0 (safe for log). */
static inline double rng_uniform(rng_t *r) {
    return ((rng_next(r) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

/* Standard normal via Box-Muller; caches the spare deviate. */
typedef struct { rng_t rng; double spare; int has_spare; } normal_t;

static inline void normal_init(normal_t *n, uint64_t seed) {
    rng_seed(&n->rng, seed); n->has_spare = 0; n->spare = 0.0;
}

static inline double normal_next(normal_t *n) {
    if (n->has_spare) { n->has_spare = 0; return n->spare; }
    double u1 = rng_uniform(&n->rng);
    double u2 = rng_uniform(&n->rng);
    double mag = sqrt(-2.0 * log(u1));
    n->spare = mag * sin(2.0 * M_PI * u2);
    n->has_spare = 1;
    return mag * cos(2.0 * M_PI * u2);
}

/* --------------------------------------------------- single-path simulation --
 * Returns the portfolio LOSS (V0 - V_T) for one Monte Carlo draw. `z` is
 * caller-provided scratch of length n_assets to avoid per-call allocation.
 */
static inline double mc_simulate_loss(const Portfolio *p, normal_t *n, double *z) {
    const int N = p->n_assets;
    const double T = p->horizon_years;
    const double sqrtT = sqrt(T);

    /* Correlated normals: z = L * x, with L lower-triangular. */
    double V_T = 0.0, V_0 = 0.0;
    double x[MC_MAX_ASSETS];
    for (int i = 0; i < N; i++) x[i] = normal_next(n);
    for (int i = 0; i < N; i++) {
        double zi = 0.0;
        const double *Li = &p->chol[i * N];
        for (int j = 0; j <= i; j++) zi += Li[j] * x[j];
        z[i] = zi;
    }
    for (int i = 0; i < N; i++) {
        double drift = (p->mu[i] - 0.5 * p->sigma[i] * p->sigma[i]) * T;
        double diff  = p->sigma[i] * sqrtT * z[i];
        double ST    = p->S0[i] * exp(drift + diff);
        V_T += p->weights[i] * ST;
        V_0 += p->weights[i] * p->S0[i];
    }
    return V_0 - V_T;
}

/* ----------------------------------------------------- shared (common.c) ----- */
int    cholesky(const double *corr, double *L, int n);   /* 0 on success */
void   portfolio_demo(Portfolio *p);                      /* built-in 5-asset */
int    portfolio_load(Portfolio *p, const char *path);    /* CSV loader */
double portfolio_V0(const Portfolio *p);
void   compute_var_cvar(double *losses, size_t n, double alpha,
                        double *var_out, double *cvar_out, double *mean_out);
void   print_result_json(const char *backend, const McResult *r);

#endif /* MONTECARLO_H */
