#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <string.h>
#include <time.h>
#include "tools.h"
#include <stdbool.h>

typedef struct 
{
    bool accepted; 
    double wave;
} result_t;

typedef struct
{
    double autocorrelation;
    double block_average;
    double alpha;
} result_mcmc;

double wave(double* r1, double* r2, double alpha);
double d_wave(double* r1, double* r2, double alpha);
void displace_electron(double* r, double delta, gsl_rng* k);
result_t variational_mcmc_one_step(double* r1, double* r2, double delta, gsl_rng* k, double alpha);
gsl_rng* get_rand(void);
result_mcmc variational_mcmc(double r1[3], double r2[3], int n, int n_eq, double alpha,
     double delta, bool adjust_alpha, double a, double beta, bool print, bool write_file);
double get_energy(double* r1, double* r2, double alpha);
void task1(void);
void task2(void);
void task3(void);
double autocorrelation(double *data, int data_len, int time_lag_ind);
double block_average(double *data, int data_len, int block_size);

int
run(
    int argc,
    char *argv[]
   )
{
    // TASK 1
    task1();
   // task3();
    return 0;
}

void task1(void)
{
    double r1[] = {0.5, 0, 0};
    double r2[] = {0, -0.5, 0};
    double alpha = 0.1;
    double delta = 2; 
    int n = 100000;
    int n_eq = 0;
    result_mcmc result = variational_mcmc(r1, r2, n, n_eq, alpha, delta, false, 1, 0.9, true, true);
}

void task2(void)
{
    double r1[] = {5, 0, 0};
    double r2[] = {0, 5, 0};
    double alpha = 0.1;
    double delta = 2; 
    int n = 100000;
    int n_eq = 20000;
    result_mcmc result = variational_mcmc(r1, r2, n, n_eq, alpha, delta, false, 1, 0.9, true, false);
}

void task3(void)
{
    double r1[] = {1, 0, 0};
    double r2[] = {0, 1, 0};
    double alpha0 = 0.05;
    double increment = 0.01;
    int n_alphas = 21;
    double alphas[n_alphas];
    double autocorrelations[n_alphas];
    double block_avgs[n_alphas];
    double delta = 2; 
    int n = 100000;
    int n_eq = 20000;
    int n_runs = 100;

    // Generating alpha values.
    for (int i = 0; i < n_alphas; i++)
    {
        alphas[i] = alpha0; 
        alpha0 += increment;
    }

    // Calculating the statistical inefficiency for each alpha value.
    for (int i = 0; i < n_alphas; i++)
    {
        double autocorrelation_accum = 0;
        double block_avg_accum = 0; 
        for (int j = 0; j < n_runs; j++)
        {
            result_mcmc result = variational_mcmc(r1, r2, n, n_eq, alphas[i],
                delta, false, 1, 1, false, false);
            autocorrelation_accum += result.autocorrelation;
            block_avg_accum += result.block_average;
        }
        autocorrelations[i] = autocorrelation_accum / n_runs; 
        block_avgs[i] = block_avg_accum / n_runs;
    }


}

result_mcmc variational_mcmc(double r1[3], double r2[3], int n, int n_eq, double alpha,
    double delta, bool adjust_alpha, double a, double beta, bool print, bool write_file)
{
    gsl_rng* k = get_rand();
    int accepted = 0;
    double energy_accum = 0;
    double d_ln_wave_accum = 0;
    double energy_wave_accum = 0;
    double* energies = (double*) malloc(sizeof(double) * n - n_eq);
    double temp_r1[3];
    double temp_r2[3];
    char buffer[50];
    sprintf(buffer, "data/data_neq%d_alpha_%.3f.csv", n_eq, alpha);
    FILE* file;
    if (write_file == true)
    {
        file = fopen(buffer, "w+");
    }
    for (int t = 0; t < n; t++)
    {
        // One step of the MCMC
        memcpy(temp_r1, r1, sizeof(temp_r1));
        memcpy(temp_r2, r2, sizeof(temp_r2));
        result_t result = variational_mcmc_one_step(temp_r1, temp_r2, delta, k, alpha);
        if (result.accepted == true)
        {   
            accepted++;
            memcpy(r1, temp_r1, sizeof(temp_r1));
            memcpy(r2, temp_r2, sizeof(temp_r2));
        }

        // Continuing to the next timestep before calculating quantities and
        // writing to file if we are in the equilibration phase. 
        if (t >= n_eq)
        {
            double energy = get_energy(r1, r2, alpha);
            double d_ln_wave = d_wave(r1, r2, alpha);
            energies[t] = energy; 
            energy_accum += energy;
            d_ln_wave_accum += d_ln_wave;
            energy_wave_accum += energy * d_ln_wave;
            if (write_file == true)
            {
                fprintf(file, "%f,%f,%f,%f,%f,%f,%f,%f\n", r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], alpha, energy);
            }
            // Adjusting alpha using gradient descent.
            if(adjust_alpha == true)
            {
                int p = t - n_eq + 1;
                double step = a * pow(p, - beta);
                double d_alpha = 2 * ((energy_wave_accum / p) - (energy_accum / p) * (d_ln_wave_accum / p));
                alpha -= step * d_alpha;
            }
        }
    }
    double block_avg = block_average(energies, n - n_eq, 1000);
    double autocor = autocorrelation(energies, n - n_eq, 1000);
    if (print == true)
    {
        printf("Fraction accepted: %.5f\nAutocorrelation: %.5f\nBlock average: %.5f\nAlpha: %.5f\n",
            (float) accepted / n, autocor, block_avg, alpha);
    }
    result_mcmc result; 
    result.alpha = alpha; 
    result.autocorrelation = autocor;
    result.block_average = block_avg;
    fclose(file);
    return result;
}

result_t variational_mcmc_one_step(double* r1, double* r2, double delta, gsl_rng* k, double alpha)
{
    result_t result;
    result.accepted = 0;
    result.wave = wave(r1, r2, alpha);
    displace_electron(r1, delta, k);
    displace_electron(r2, delta, k);
    double w2 = wave(r1, r2, alpha);
    double r = gsl_rng_uniform(k);
    if (r < (w2 * w2)/ (result.wave * result.wave))
    {
        result.wave = w2;
        result.accepted = 1;
    }
    return result; 
}

void displace_electron(double* r, double delta, gsl_rng* k)
{
    for (size_t i = 0; i < 3; i++)
    {
        r[i] += (gsl_rng_uniform(k) - 0.5) * delta;
    }
}

double wave(double* r1, double* r2, double alpha)
{   
    double r1_len = vector_norm(r1, 3);
    double r2_len = vector_norm(r2, 3);
    double r12_len = distance_between_vectors(r1, r2, 3);
    return exp(- 2 * r1_len) * exp(- 2 * r2_len) * 
        exp(r12_len / (2 + 2 * alpha * r12_len));
}

double d_wave(double* r1, double* r2, double alpha)
{
    double r12_len = distance_between_vectors(r1, r2, 3);
    return 2 * r12_len / pow(2 * r12_len * alpha + 2, 2); 
}

double get_energy(double* r1, double* r2, double alpha)
{
    double r12_len = distance_between_vectors(r1, r2, 3);
    double r_norm_diff[] = {0, 0, 0};
    double r_diff[] = {0, 0, 0};
    double r1_norm[3]; 
    double r2_norm[3];
    memcpy(r1_norm, r1, sizeof(r1_norm)); 
    memcpy(r2_norm, r2, sizeof(r2_norm)); 
    double denominator = 1 + alpha * r12_len;
    normalize_vector(r1_norm, 3);
    normalize_vector(r2_norm, 3);
    elementwise_subtraction(r_norm_diff, r1_norm, r2_norm, 3);
    elementwise_subtraction(r_diff, r1, r2, 3);
    return - 4 * (dot_product(r_norm_diff, r_diff, 3)) / (r12_len * pow(denominator, 2)) -
        1 / (r12_len * pow(denominator, 3)) - 1 / (4 * pow(denominator, 4)) + 1 / r12_len;
}

gsl_rng* get_rand(void){
    const gsl_rng_type* T;
    gsl_rng* r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    time_t seed = time(NULL);
    gsl_rng_set(r, seed);
    return r;
}

double autocorrelation(double *data, int data_len, int time_lag_ind)
{
    double mean = average(data, data_len);
    double var = variance(data, data_len);
    double cov = 0;
    for (int idx = 0; idx < data_len - time_lag_ind; idx++)
    {
        cov += (data[idx] - mean) * (data[idx + time_lag_ind] - mean); // Covariance 
    }
    return cov / (var * (data_len - time_lag_ind)); // Covariance / variance = correlation
}

double block_average(double *data, int data_len, int block_size)
{
    int n_blocks = data_len / block_size;
    double* blocks = (double*) malloc(sizeof(double) * n_blocks);
    memset(blocks, 0, n_blocks * sizeof(double));
    for (int jdx = 0; jdx < n_blocks; jdx++)
    {
        for (int idx = 0; idx < block_size; idx++)
        {
            blocks[jdx] += data[idx + jdx * block_size];
        }
        blocks[jdx] /= block_size;
    }
    double var = variance(data, data_len);
    double block_var = variance(blocks, n_blocks);
    free(blocks);
    return block_size * block_var / var;
}
