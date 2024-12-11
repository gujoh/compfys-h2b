#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <string.h>
#include <time.h>
#include "tools.h"

double wave(double* r1, double* r2, double alpha);
void displace_electron(double* r, double delta, gsl_rng* k);
int variational_mcmc_one_step(double* r1, double* r2, double delta, gsl_rng* k, double alpha);
gsl_rng* get_rand(void);
void variational_mcmc(void);
double get_energy(double* r1, double* r2, double alpha);

int
run(
    int argc,
    char *argv[]
   )
{
    variational_mcmc();
    return 0;
}

void variational_mcmc(void)
{
    double r1[] = {2, 0, 0};
    double r2[] = {-2, 0, 0};
    double alpha = 0.1;
    double delta = 2;
    gsl_rng* k = get_rand();
    int accepted = 0;
    int n = 100000;
    double temp_r1[3];
    double temp_r2[3];
    FILE* file = fopen("data/positions.csv", "w+");
    if (file == NULL)
    {
        printf("File could not be opened.\n");
        exit(1); 
    }
    for (int i = 0; i < n; i++)
    {
        memcpy(temp_r1, r1, sizeof(temp_r1));
        memcpy(temp_r2, r2, sizeof(temp_r2));
        int result = variational_mcmc_one_step(temp_r1, temp_r2, delta, k, alpha);
        if (result == 1)
        {   
            accepted++;
            memcpy(r1, temp_r1, sizeof(r1));
            memcpy(r2, temp_r2, sizeof(r2));
        }
        double energy = get_energy(r1, r2, alpha);
        fprintf(file, "%f,%f,%f,%f,%f,%f,%f\n", r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], energy);
    }
    printf("Fraction accepted: %.4f\n", (float) accepted / n);
    fclose(file);
}

int variational_mcmc_one_step(double* r1, double* r2, double delta, gsl_rng* k, double alpha)
{
    double w1 = wave(r1, r2, alpha);
    displace_electron(r1, delta, k);
    displace_electron(r2, delta, k);
    double w2 = wave(r1, r2, alpha);
    double r = gsl_rng_uniform(k);
    if (r < w2 / w1)
    {
        return 1;  
    }
    return 0; 
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
