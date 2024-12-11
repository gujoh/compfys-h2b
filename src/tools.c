#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <complex.h>
#include <string.h>
#include "tools.h"

void
elementwise_addition(
                     double *res,
                     double *v1,
                     double *v2,
                     unsigned int len
                    )
{
    for (size_t idx = 0; idx < len; idx++)
    {
        res[idx] = v1[idx] + v2[idx];
    }
}

void
elementwise_subtraction(
                     double *res,
                     double *v1,
                     double *v2,
                     unsigned int len
                    )
{
    for (size_t idx = 0; idx < len; idx++)
    {
        res[idx] = v1[idx] - v2[idx];
    }
}

void
elementwise_multiplication(
                           double *res,
                           double *v1,
                           double *v2,
                           unsigned int len
                          )
{
    for (size_t idx = 0; idx < len; idx++)
    {
        res[idx] = v1[idx] * v2[idx];
    }
}

void
addition_with_constant(
                       double *res,
                       double *v,
                       double constant,
                       unsigned int len)
{
for (size_t idx = 0; idx < len; idx++)
    {
        res[idx] = v[idx] + constant;
    }
}

void
multiplication_with_constant(
                             double *res,
                             double *v,
                             double constant,
                             unsigned int len)
{
for (size_t idx = 0; idx < len; idx++)
    {
        res[idx] = v[idx] * constant;
    }
}

double
dot_product(
            double *v1,
            double *v2,
            unsigned int len
           )
{
    double res = 0;
    for (size_t idx = 0; idx < len; idx++)
    {
        res += v1[idx] * v2[idx];
    }
    return res;
}

double **
create_2D_array(
                unsigned int row_size,
                unsigned int column_size
               )
{
    double* asentries = (double*) malloc(row_size * column_size * sizeof(double));
    double** as = (double**) malloc(row_size * sizeof(double*));
    for (size_t idx = 0; idx < row_size; idx++)
    {
        //array[idx] = (double*) malloc(sizeof(double) * column_size);
        as[idx] = asentries + column_size * idx;  
    }
    return as;
}

void
destroy_2D_array(
                 double **array
                )

{
    free(array[0]);
    free(array);
}

void
matrix_vector_multiplication(
                             double *result,
                             double **A,
                             double *b,
                             unsigned int n,
                             unsigned int m
                            )
{
    for (size_t idx = 0; idx < n; idx++)
    {
        result[idx] = 0;
        for (size_t jdx = 0; jdx < m; jdx++)
        {
            result[idx] += A[idx][jdx] * b[jdx];
        }
    }
}

void
matrix_matrix_multiplication(
                             double **result,
                             double **A,
                             double **B,
                             unsigned int n,
                             unsigned int m,
                             unsigned int k
                            )
{
    for (size_t idx = 0; idx < n; idx++)
    {
        for (size_t jdx = 0; jdx < k; jdx++)
        {
            result[idx][jdx] = 0;
            for (size_t kdx = 0; kdx < m; kdx++)
            {
                result[idx][jdx] += A[idx][kdx] * B[kdx][jdx];
            }
        }
    }
}

double
vector_norm(
            double *v1,
            unsigned int len
           )
{
    double result = 0;
    for (size_t idx = 0; idx < len; idx++)
    {
        result += v1[idx] * v1[idx];
    }
    return sqrt(result);
}


void
normalize_vector(
                 double *v1,
                 unsigned int len
                )
{
    double norm = vector_norm(v1, len);
    for (size_t idx = 0; idx < len; idx++)
    {
        v1[idx] /= norm;
    }
}

double
average(
        double *v1,
        unsigned int len
       )
{
    double res = 0;
    for (size_t idx = 0; idx < len; idx++)
    {
        res += v1[idx];
    }
    return res / len;
}

double variance(
                    double *v1,
                    unsigned int len
)
{
    double result = 0;
    double mean = average(v1, len);
    for (size_t idx = 0; idx < len; idx++)
    {
        double delta = v1[idx] - mean;
        result += delta * delta;
    }
    return result / len;
}

double
standard_deviation(
                       double *v1,
                       unsigned int len
                  )
{
    return sqrt(variance(v1, len));
}

double
distance_between_vectors(
                         double *v1,
                         double *v2,
                         unsigned int len
                        )
{
    double result = 0;
    for (size_t idx = 0; idx < len; idx++)
    {
        double delta = v1[idx] - v2[idx];
        result += delta * delta;
    }
    return sqrt(result);
}

void
cumulative_integration(
                       double *res,
                       double *v,
                       double dx,
                       unsigned int v_len
                      )
{
    res[0] = v[0];
    for (size_t idx = 1; idx < v_len; idx++)
    {
        res[idx] = res[idx-1] + ((v[idx-1] + v[idx]) / 2) * dx;
    }
}


void
write_xyz(
          FILE *fp,
          char *symbol,
          double **positions,
          double **velocities,
          double alat,
          int natoms)
{
    fprintf(fp, "%i\nLattice=\"%f 0.0 0.0 0.0 %f 0.0 0.0 0.0 %f\" ", natoms, alat, alat, alat);
    fprintf(fp, "Properties=species:S:1:pos:R:3:vel:R:3 pbc=\"T T T\"\n");
    for(int i = 0; i < natoms; ++i){
        fprintf(fp, "%s %f %f %f %f %f %f\n", symbol, 
            positions[i][0], positions[i][1], positions[i][2],
            velocities[i][0], velocities[i][1], velocities[i][2]);
    }
}

void fft_freq(
          double *res,
              int n,
              double timestep)
{
    int m = n + 1;
    int mid;
    if (m % 2 == 0)
    {
        mid = m / 2 - 1;
        for (int idx = 0; idx <= mid; idx++)
        {
            res[idx] = 2 * M_PI * idx / (timestep * n);
        }
        for (int idx = mid + 1, jdx = - mid + 1; idx < n; idx++, jdx++)
        {
            res[idx] = 2 * M_PI * jdx / (timestep * n);
        }
    }
    else 
    {
        mid = (m - 1) / 2;
        for (int idx = 0; idx <= mid; idx++)
        {
            res[idx] = 2 * M_PI * idx / (timestep * n);
        }
        for (int idx = mid + 1, jdx = - mid + 1; idx < n; idx++, jdx++)
        {
            res[idx] = 2 * M_PI * jdx / (timestep * n);
        }
    }
}

/* Freely given functions */
void
skip_line(FILE *fp)
{
    int c;
    while (c = fgetc(fp), c != '\n' && c != EOF);
}

void
read_xyz(
         FILE *fp,
         char *symbol,
         double **positions,
         double **velocities,
         double *alat)
{
    int natoms;
    if(fscanf(fp, "%i\nLattice=\"%lf 0.0 0.0 0.0 %lf 0.0 0.0 0.0 %lf\" ", &natoms, alat, alat, alat) == 0){
        perror("Error");
    }
    skip_line(fp);
    for(int i = 0; i < natoms; ++i){
        fscanf(fp, "%s %lf %lf %lf ",
                symbol, &positions[i][0], &positions[i][1], &positions[i][2]);
        fscanf(fp, "%lf %lf %lf\n",
                &velocities[i][0], &velocities[i][1], &velocities[i][2]);
    }
}

void powerspectrum(
           double *res,
           double *signal,
           int n,
                   double timestep)
{
    /* Declaration of variables */
    double *complex_coefficient = malloc(sizeof(double) * 2*n); // array for the complex fft data
    double *data_cp = malloc(sizeof(double) * n);

    /*make copy of data to avoid messing with data in the transform*/
    for (int i = 0; i < n; i++) {
    data_cp[i] = signal[i];
    }

    /* Declare wavetable and workspace for fft */
    gsl_fft_real_wavetable *real;
    gsl_fft_real_workspace *work;

    /* Allocate space for wavetable and workspace for fft */
    work = gsl_fft_real_workspace_alloc(n);
    real = gsl_fft_real_wavetable_alloc(n);

    /* Do the fft*/
    gsl_fft_real_transform(data_cp, 1, n, real, work);

    /* Unpack the output into array with alternating real and imaginary part */
    gsl_fft_halfcomplex_unpack(data_cp, complex_coefficient,1,n);

    /*fill the output powspec_data with the powerspectrum */
    for (int i = 0; i < n; i++) {
    res[i] = (complex_coefficient[2*i]*complex_coefficient[2*i]+complex_coefficient[2*i+1]*complex_coefficient[2*i+1]);
    res[i] *= timestep / n;
    }

    /* Free memory of wavetable and workspace */
    gsl_fft_real_wavetable_free(real);
    gsl_fft_real_workspace_free(work);
    free(complex_coefficient);
    free(data_cp);
}
