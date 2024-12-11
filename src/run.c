#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <string.h>

typedef struct
{
    int accepted;
    int wave; 
} result_t; 

double wave(double* r1, double* r2, double alpha);
void displace_electron(double* r, double delta, gsl_rng* k);
result_t variational_mcmc_one_step(result_t result, double* r1, double* r2, double delta, gsl_rng* k, double alpha);

int
run(
    int argc,
    char *argv[]
   )
{
    printf("Hello, world!\n");
    return 0;
}

void variational_mcmc()
{
    
}

result_t variational_mcmc_one_step(result_t result, double* r1, double* r2, double delta, gsl_rng* k, double alpha)
{
    result.accepted = 0;
    displace_electron(r1, delta, k);
    displace_electron(r2, delta, k);
    double w2 = wave(r1, r2, alpha);
    double r = gsl_rng_uniform(k);
    if (r < w2 / result.wave)
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