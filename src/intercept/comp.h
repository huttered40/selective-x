#ifndef SELECTIVEX__INTERCEPT__COMP_H_
#define SELECTIVEX__INTERCEPT__COMP_H_

// C interface
// BLAS 1
void selectivex_daxpy_(const int n , const double a , const double *x , const int incx , double *y , const int incy);
void selectivex_dscal_(const int n , const double a , double *x , const int incx);

// BLAS 2
void selectivex_dgbmv_(const int order, const int trans, const int m, const int n, const int kl, const int ku, const double alpha,
             const double *a, const int lda, const double *x, const int incx, const double beta, double *y, const int incy);
void selectivex_dgemv_(const int order, const int trans , const int m , const int n, const double alpha , const double *a ,
             const int lda , const double *x, const int incx , const double beta, double *y , const int incy );
void selectivex_dger_(const int order, const int m , const int n , const double alpha , const double *x , const int incx ,
            const double *y , const int incy , double *a , const int lda);
void selectivex_dsbmv_(const int Layout, const int uplo, const int n, const int k, const double alpha, const double *a,
             const int lda, const double *x, const int incx, const double beta, double *y, const int incy);
void selectivex_dspmv_(const int Layout, const int uplo, const int n, const double alpha, const double *ap, const double *x,
             const int incx, const double beta, double *y, const int incy);
void selectivex_dspr_(const int Layout, const int uplo, const int n, const double alpha, const double *x,
            const int incx, double *ap);
void selectivex_dspr2_(const int Layout, const int uplo, const int n, const double alpha, const double *x, const int incx,
             const double *y, const int incy, double *ap);
void selectivex_dsymv_(const int Layout, const int uplo, const int n, const double alpha, const double *a, const int lda,
            const double *x, const int incx, const double beta, double *y, const int incy);
void selectivex_dsyr_(const int Layout, const int uplo, const int n, const double alpha, const double *x, const int incx,
            double *a, const int lda);
void selectivex_dsyr2_(const int Layout, const int uplo, const int n, const double alpha, const double *x, const int incx,
             const double *y, const int incy, double *a, const int lda);
void selectivex_dtrsv_(const int order, const int uplo, const int trans, const int diag, const int n, const double *a,
             const int lda, double *x, const int incx);
void selectivex_dtrmv_(const int order, const int uplo , const int trans , const int diag , const int n , const double *a ,
             const int lda , double *x, const int incx );
void selectivex_dtpsv_(const int order, const int uplo, const int trans, const int diag, const int n, const double *ap,
             double *x, const int incx);
void selectivex_dtpmv_(const int order, const int uplo, const int trans, const int diag, const int n, const double *ap,
             double *x, const int incx);
void selectivex_dtbsv_(const int order, const int uplo, const int trans, const int diag, const int n, const int k,
             const double *a, const int lda, double *x, const int incx);
void selectivex_dtbmv_(const int order, const int uplo, const int trans, const int diag, const int n, const int k,
             const double *a, const int lda, double *x, const int incx);

// BLAS 3
void selectivex_dgemm_(const int order, const int transa , const int transb ,
             const int m , const int n , const int k , const double alpha , const double *a ,
             const int lda , const double *b , const int ldb , const double beta , double *c , const int ldc);
void selectivex_dtrmm_(const int order, const int side , const int uplo , const int transa ,
             const int diag , const int m , const int n , const double alpha , const double *a ,
             const int lda , double *b , const int ldb);
void selectivex_dtrsm_(const int order, const int side , const int uplo , const int transa ,
             const int diag , const int m , const int n , const double alpha , const double *a ,
             const int lda , double *b , const int ldb);
void selectivex_dsyrk_(const int order, const int uplo , const int trans ,
             const int n , const int k , const double alpha , const double *a , const int lda ,
             const double beta , double *c , const int ldc);
void selectivex_dsyr2k_(const int order, const int uplo, const int trans, const int n, const int k, const double alpha,
              const double *a, const int lda, const double *b, const int ldb, const double beta, double *c,
              const int ldc);
void selectivex_dsymm_(const int order, const int side, const int uplo, const int m, const int n, const double alpha,
             const double *a, const int lda, const double *b, const int ldb, const double beta, double *c,
             const int ldc);

// FORTRAN interface
// BLAS 1
void selectivex__daxpy__(const int* n , const double* a , const double *x , const int* incx , double *y , const int* incy);
void selectivex__dscal__(const int* n , const double* a , double *x , const int* incx);

// BLAS 2
void selectivex__dgbmv__(const char* trans , const int* m , const int* n, const int* kl, const int* ku, const double* alpha ,
               const double *a , const int* lda , const double *x, const int* incx ,
               const double* beta, double *y , const int* incy );
void selectivex__dgemv__(const char* trans , const int* m , const int* n, const double* alpha , const double *a ,
               const int* lda , const double *x, const int* incx ,
               const double* beta, double *y , const int* incy );
void selectivex__dger__(const int* m , const int* n , const double* alpha , const double *x , const int* incx ,
              const double *y , const int* incy , double *a , const int* lda);
void selectivex__dsbmv__(const char* uplo, const int* n, const int* k, const double* alpha, const double *a,
               const int* lda, const double *x, const int* incx, const double* beta, double *y, const int* incy);
void selectivex__dspmv__(const char* uplo, const int* n, const double* alpha, const double *ap, const double *x,
               const int* incx, const double* beta, double *y, const int* incy);
void selectivex__dspr__(const char* uplo, const int* n, const double* alpha, const double *x,
              const int* incx, double *ap);
void selectivex__dspr2__(const char* uplo, const int* n, const double* alpha, const double *x, const int* incx,
               const double *y, const int* incy, double *ap);
void selectivex__dsymv__(const char* uplo, const int* n, const double* alpha, const double *a, const int* lda,
               const double *x, const int* incx, const double* beta, double *y, const int* incy);
void selectivex__dsyr__(const char* uplo, const int* n, const double* alpha, const double *x, const int* incx,
              double *a, const int* lda);
void selectivex__dsyr2__(const char* uplo, const int* n, const double* alpha, const double *x, const int* incx,
               const double *y, const int* incy, double *a, const int* lda);
void selectivex__dtrsv__(const char* uplo, const char* trans, const char* diag, const int* n, const double *a,
               const int* lda, double *x, const int* incx);
void selectivex__dtrmv__(const char* uplo , const char* trans , const char* diag , const int* n , const double *a ,
               const int* lda , double *x, const int* incx );
void selectivex__dtpsv__(const char* uplo, const char* trans, const char* diag, const int* n, const double *ap,
               double *x, const int* incx);
void selectivex__dtpmv__(const char* uplo, const char* trans, const char* diag, const int* n, const double *ap,
               double *x, const int* incx);
void selectivex__dtbsv__(const char* uplo, const char* trans, const char* diag, const int* n, const int* k,
               const double *a, const int* lda, double *x, const int* incx);
void selectivex__dtbmv__(const char* uplo, const char* trans, const char* diag, const int* n, const int* k,
               const double *a, const int* lda, double *x, const int* incx);

// BLAS 3
void selectivex__dgemm__(const char* transa , const char* transb ,
               const int* m , const int* n , const int* k , const double* alpha , const double *a ,
               const int* lda , const double *b , const int* ldb , const double* beta , double *c , const int* ldc);
void selectivex__dtrmm__(const char* side , const char* uplo , const char* transa ,
               const char* diag , const int* m , const int* n , const double* alpha , const double *a ,
               const int* lda , double *b , const int* ldb);
void selectivex__dtrsm__(const char* side , const char* uplo , const char* transa ,
               const char* diag , const int* m , const int* n , const double* alpha , const double *a ,
               const int* lda , double *b , const int* ldb);
void selectivex__dsyrk__(const char* uplo, const char* trans ,
               const int* n , const int* k , const double* alpha , const double *a , const int* lda ,
               const double* beta , double *c , const int* ldc);
void selectivex__dsyr2k__(const char* uplo, const char* trans, const int* n, const int* k, const double* alpha,
                const double *a, const int* lda, const double *b, const int* ldb, const double* beta, double *c,
                const int* ldc);
void selectivex__dsymm__(const char* side, const char* uplo, const int* m, const int* n, const double* alpha,
               const double *a, const int* lda, const double *b, const int* ldb, const double* beta, double *c,
               const int* ldc);

// C interface
int selectivex_dgetrf_(int matrix_layout, int m , int n , double* a , int lda , int* ipiv);
int selectivex_dpotrf_(int matrix_layout, char uplo , int n , double* a , int lda);
int selectivex_dtrtri_(int matrix_layout, char uplo , char diag , int n , double* a , int lda);
int selectivex_dgeqrf_(int matrix_layout, int m , int n , double* a , int lda , double* tau);
int selectivex_dorgqr_(int matrix_layout, int m , int n , int k , double* a , int lda , const double* tau);
int selectivex_dormqr_(int matrix_layout, char side , char trans , int m , int n , int k , const double * a , int lda , const double * tau , double * c , int ldc);
int selectivex_dgetri_(int matrix_layout, int n , double * a , int lda , const int * ipiv);
int selectivex_dtpqrt_(int matrix_layout, int m , int n , int l , int nb , double * a , int lda , double * b , int ldb , double * t , int ldt);
int selectivex_dtpmqrt_(int matrix_layout, char side , char trans , int m , int n , int k , int l , int nb , const double * v ,
               int ldv , const double * t , int ldt , double * a , int lda , double * b , int ldb);

// FORTRAN interface
void selectivex__dgetrf__(const int* m , const int* n , double* a , const int* lda , int* ipiv, int* info);
void selectivex__dpotrf__(const char* uplo , const int* n , double* a , const int* lda, int* info);
void selectivex__dtrtri__(const char* uplo , const char* diag , const int* n , double* a , const int* lda, int* info);
void selectivex__dgeqrf__(const int* m , const int* n , double* a , const int* lda , double* tau, double* work, const int* lwork, int* info);
void selectivex__dorgqr__(const int* m , const int* n , const int* k , double* a , const int* lda , const double* tau, double* work, const int* lwork, int* info);
void selectivex__dormqr__(const char* side , const char* trans , const int* m , const int* n , const int* k , const double * a , const int* lda , const double * tau ,
                double * c , const int* ldc, double* work, const int* lwork, int* info);
void selectivex__dgetri__(const int* n , double * a , const int* lda , const int * ipiv, double* work, const int* lwork, int* info);
void selectivex__dtpqrt__(const int* m , const int* n , const int* l , const int* nb , double * a , const int* lda , double* b , const int* ldb , double * t , const int* ldt,
                double* work, int* info);
void selectivex__dtpmqrt__(const char* side , const char* trans , const int* m , const int* n , const int* k , const int* l , const int* nb , const double * v ,
                 const int* ldv , const double * t , const int* ldt , double * a , const int* lda , double * b , const int* ldb, double* work, int* info);

#include "func_generators.h"

#endif /*SELECTIVEX__INTERCEPT__COMP_H_*/
