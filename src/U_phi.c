#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 /*#include <R.h>*/
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <omp.h> 


void mixpdf(const int *nr, const int *K, const int *nl, double *pi, double *x, double *mu, double *Sigma,double *assignment);
void gsl_matrix_set_zero (gsl_matrix * m);
gsl_matrix * gsl_matrix_alloc (size_t n1, size_t n2);
void gsl_matrix_set (gsl_matrix * m, size_t i, size_t j, double x);
double dmvnorm(const int n, const gsl_vector *x, const gsl_vector *mean, const gsl_matrix *var);
size_t gsl_vector_max_index (const gsl_vector * v);
void gsl_vector_free (gsl_vector * v);
void inner(double *a, double *b, const int *num_elements, double *res);
void row_prod(double *a, const int *num_row, const int *num_col, double *res);
void U_phi(double *V, double *B, const int *N, const int *n, const int *q, const int *D, double *M, 
double *tmp1, double *tmp2, double *v, double*b, double *res);
void M_func(double *V, double *B, const int *N, const int *n, const int *q, const int *D, double *v, double*b, double *M,  double *tmp2, double *res);
double dot_product(double *v, double *u, const int *n);

double dot_product(double *v, double *u, const int *n)
{
    double result = 0.0;
    for (int i = 0; i < *n; i++)
        result += (*(v+i)) * (*(u+i));
    return result;
}


void inner(double *a, double *b, const int *num_elements, double *res){
   for (int i=0; i < *num_elements; ++i){
   *res += (*(a+i)) * (*(b+i));
  }
}

void row_prod(double *a, const int *num_row, const int *num_col, double *res){
  for (int i=0; i< *num_row; ++i) {
    *(res+i) = 1;
    for (int j=0; j < *num_col; ++j) {
      *(res+i) *= *(a+j*(*num_row) + i);
    }
  }
}


void U_phi(double *V, double *B, const int *N, const int *n, const int *q, const int *D, double *M, 
double *tmp1, double *tmp2, double *v, double*b, double *res){
  
  for (int i=0; i< *N; ++i) {
/*  double *M;
   M = (double *) malloc((*q)*(*D));
     double *M = NULL,*tmp1 = NULL,*tmp2 = NULL; */
   for (int j=0; j< *q; ++j) {
      for (int k=0; k< *D; ++k) {
         *(M + k*(*q) + j) = 0;
      }
    }
  
   for (int j=0; j< *q; ++j) {
      for (int k=0; k< *D; ++k) {
        
       /* double *v= NULL,*b= NULL;  */
        for (int m=0; m < *n; ++m) {
          *(v+m) = *(V + k*(*n)*(*q) + j*(*n) +m);
          *(b+m) = *(B + k*(*n)*(*N) + i*(*n) +m);
        }

        inner(v, b, n, tmp1);
        *(M + k*(*q) + j) = *tmp1; 
      }
        row_prod(M, q, D, tmp2);
      *(res + i*(*q) + j) = *(tmp2+j); 
    } 
  } 
}


void M_func(double *V, double *B, const int *N, const int *n, const int *q, const int *D, double *v, double*b, double *M,  double *tmp2, double *res){

/*  #pragma omp parallel for    */

for (int i=0; i< *N; ++i) {

  for (int j=0; j< *q; ++j) {
  
     for (int k=0; k< *D; ++k) {
        for (int m=0; m < *n; ++m) {
          *(v+m) = *(V + k*(*n)*(*q) + j*(*n) +m);
          *(b+m) = *(B + k*(*n)*(*N) + i*(*n) + m);
        }
       *(M + k*(*q) + j ) = dot_product(v, b, n);
      }
  }
  
    for (int j=0; j< *q; ++j) {
          row_prod(M, q, D, tmp2);
        *(res  + i*(*q) + j) = *(tmp2+j); 
    }


}
}













  