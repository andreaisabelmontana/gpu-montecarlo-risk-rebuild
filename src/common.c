/* common.c — backend-independent helpers: Cholesky, portfolio setup,
 * VaR/CVaR estimation, and result reporting.
 */
#include "montecarlo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Cholesky factorization of an SPD correlation matrix: corr = L * L^T.
 * Writes the lower-triangular L (row-major, n x n). Returns 0 on success,
 * -1 if the matrix is not positive-definite (a non-positive pivot). */
int cholesky(const double *corr, double *L, int n) {
    memset(L, 0, sizeof(double) * (size_t)n * (size_t)n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = corr[i * n + j];
            for (int k = 0; k < j; k++) sum -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (sum <= 0.0) return -1;   /* not positive-definite */
                L[i * n + j] = sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }
    return 0;
}

/* A built-in, realistic 5-asset portfolio (equities + bond-like asset). */
void portfolio_demo(Portfolio *p) {
    const int N = 5;
    p->n_assets = N;
    p->horizon_years = 1.0;

    double S0[]     = {100, 120,  80,  60, 100};
    double mu[]     = {0.08, 0.10, 0.06, 0.05, 0.03};
    double sigma[]  = {0.20, 0.28, 0.18, 0.30, 0.08};
    double weights[]= {  10,    8,   15,    5,   20};
    /* Symmetric positive-definite correlation matrix. */
    double corr[5 * 5] = {
        1.00, 0.55, 0.40, 0.30, -0.10,
        0.55, 1.00, 0.35, 0.45, -0.05,
        0.40, 0.35, 1.00, 0.25,  0.00,
        0.30, 0.45, 0.25, 1.00,  0.05,
       -0.10,-0.05, 0.00, 0.05,  1.00,
    };
    memcpy(p->S0, S0, sizeof S0);
    memcpy(p->mu, mu, sizeof mu);
    memcpy(p->sigma, sigma, sizeof sigma);
    memcpy(p->weights, weights, sizeof weights);
    memcpy(p->corr, corr, sizeof corr);

    if (cholesky(p->corr, p->chol, N) != 0) {
        fprintf(stderr, "FATAL: demo correlation matrix not positive-definite\n");
        exit(1);
    }
}

/* CSV loader: first line `n_assets,horizon_years`; then n rows of
 * `S0,mu,sigma,weight`; then n rows of the correlation matrix. */
int portfolio_load(Portfolio *p, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int N; double T;
    if (fscanf(f, "%d,%lf", &N, &T) != 2 || N < 1 || N > MC_MAX_ASSETS) { fclose(f); return -1; }
    p->n_assets = N; p->horizon_years = T;
    for (int i = 0; i < N; i++)
        if (fscanf(f, "%lf,%lf,%lf,%lf", &p->S0[i], &p->mu[i], &p->sigma[i], &p->weights[i]) != 4) {
            fclose(f); return -1;
        }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (fscanf(f, "%lf%*c", &p->corr[i * N + j]) != 1) { fclose(f); return -1; }
    fclose(f);
    return cholesky(p->corr, p->chol, N);
}

double portfolio_V0(const Portfolio *p) {
    double v = 0.0;
    for (int i = 0; i < p->n_assets; i++) v += p->weights[i] * p->S0[i];
    return v;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Exact VaR/CVaR by sorting the loss sample.
 * VaR_alpha  = the alpha-quantile of the loss distribution.
 * CVaR_alpha = mean loss in the worst (1-alpha) tail. */
void compute_var_cvar(double *losses, size_t n, double alpha,
                      double *var_out, double *cvar_out, double *mean_out) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += losses[i];
    *mean_out = sum / (double)n;

    qsort(losses, n, sizeof(double), cmp_double);
    size_t idx = (size_t)(alpha * (double)n);
    if (idx >= n) idx = n - 1;
    *var_out = losses[idx];

    double tail = 0.0;
    size_t cnt = 0;
    for (size_t i = idx; i < n; i++) { tail += losses[i]; cnt++; }
    *cvar_out = cnt ? tail / (double)cnt : losses[n - 1];
}

void print_result_json(const char *backend, const McResult *r) {
    printf("{\"backend\":\"%s\",\"n_sims\":%zu,\"threads\":%d,"
           "\"seconds\":%.6f,\"sims_per_sec\":%.1f,\"V0\":%.4f,"
           "\"mean_loss\":%.4f,\"alpha\":%.4f,\"VaR\":%.4f,\"CVaR\":%.4f}\n",
           backend, r->n_sims, r->threads, r->seconds, r->sims_per_sec,
           r->V0, r->mean_loss, r->alpha, r->var, r->cvar);
}
