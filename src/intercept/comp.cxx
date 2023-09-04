#ifdef SELECTIVEX__USE_CBLAS_INTERFACE
// Note: this MKL inclusion should be conditional on config.mk
// If MKL is not used, user must specify the right header file here.
#include "mkl.h"
#define CBLAS_INT
#elif SELECTIVEX__USE_CBLAS_INTERFACE_INDI
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#define CBLAS_LAYOUT int
#define CBLAS_SIDE int
#define CBLAS_DIAG int
#define CBLAS_TRANSPOSE int
#define CBLAS_UPLO int
#endif /* SELECTIVEX__USE_CBLAS_INTERFACE */

#include <tuple>

#include "comp.h"
#include "../util.h"
#include "../interface.h"

// BLAS 1
void selectivex_daxpy_(const int n , const double a , const double *x , const int incx , double *y , const int incy){
  selectivex::selective_blas("daxpy",std::make_tuple(n),std::make_tuple(n),selectivex::IndexPack<0>{},
                          &cblas_daxpy,n,a,x,incx,y,incy);
}
void selectivex_dscal_(const int n , const double a , double *x , const int incx){
  selectivex::selective_blas("dscal",std::make_tuple(n),std::make_tuple(n),selectivex::IndexPack<0>{},
                          &cblas_dscal,n,a,x,incx);
}

// BLAS 2
void selectivex_dgbmv_(const int order, const int trans, const int m, const int n, const int kl, const int ku, const double alpha,
             const double *a, const int lda, const double *x, const int incx, const double beta, double *y, const int incy){
  selectivex::selective_blas("dgbmv",std::make_tuple(m,n,kl,ku,trans*1000+(alpha!=0)*2+(beta!=0)),
                          std::make_tuple(m,n,kl,ku,trans*1000+(alpha!=0)*2+(beta!=0)),selectivex::IndexPack<0,1,2,3,4>{},
                          &cblas_dgbmv,(CBLAS_ORDER)order,(CBLAS_TRANSPOSE)trans,m,n,kl,ku,alpha,a,lda,x,incx,beta,y,incy);
}
void selectivex_dgemv_(const int order, const int trans , const int m , const int n, const double alpha , const double *a , const int lda , const double *x, const int incx ,
             const double beta, double *y , const int incy ){
  selectivex::selective_blas("dgemv",std::make_tuple(m,n,trans*1000+(alpha!=0)*2+(beta!=0)),
                          std::make_tuple(m,n,trans*1000+(alpha!=0)*2+(beta!=0)),selectivex::IndexPack<0,1,2>{},
                          &cblas_dgemv,(CBLAS_ORDER)order,(CBLAS_TRANSPOSE)trans,m,n,alpha,a,lda,x,incx,beta,y,incy);
}
void selectivex_dger_(const int order, const int m , const int n , const double alpha , const double *x , const int incx ,
            const double *y , const int incy , double *a , const int lda){
  selectivex::selective_blas("dger",std::make_tuple(m,n,(alpha!=0)),std::make_tuple(m,n,(alpha!=0)),selectivex::IndexPack<0,1,2>{},
                          &cblas_dger,(CBLAS_ORDER)order,m,n,alpha,x,incx,y,incy,a,lda);
}
void selectivex_dsbmv_(const int Layout, const int uplo, const int n, const int k, const double alpha, const double *a,
             const int lda, const double *x, const int incx, const double beta, double *y, const int incy){
  selectivex::selective_blas("dsbmv",std::make_tuple(n,k,uplo,(alpha!=0)*2+(beta!=0)),
                          std::make_tuple(n,k,uplo,(alpha!=0)*2+(beta!=0)),selectivex::IndexPack<0,1,2,3>{},
                          &cblas_dsbmv,(CBLAS_ORDER)Layout,(CBLAS_UPLO)uplo,n,k,alpha,a,lda,x,incx,beta,y,incy);
}
void selectivex_dspmv_(const int Layout, const int uplo, const int n, const double alpha, const double *ap, const double *x,
             const int incx, const double beta, double *y, const int incy){
  selectivex::selective_blas("dspmv",std::make_tuple(n,uplo,(alpha!=0)*2+(beta!=0)),
                          std::make_tuple(n,uplo,(alpha!=0)*2+(beta!=0)),selectivex::IndexPack<0,1,2>{},
                          &cblas_dspmv,(CBLAS_ORDER)Layout,(CBLAS_UPLO)uplo,n,alpha,ap,x,incx,beta,y,incy);
}
void selectivex_dspr_(const int Layout, const int uplo, const int n, const double alpha, const double *x,
            const int incx, double *ap){
  selectivex::selective_blas("dspr",std::make_tuple(n,uplo,(alpha!=0)),std::make_tuple(n,uplo,(alpha!=0)),selectivex::IndexPack<0,1,2>{},
                          &cblas_dspr,(CBLAS_ORDER)Layout,(CBLAS_UPLO)uplo,n,alpha,x,incx,ap);
}
void selectivex_dspr2_(const int Layout, const int uplo, const int n, const double alpha, const double *x, const int incx,
             const double *y, const int incy, double *ap){
  selectivex::selective_blas("dspr2",std::make_tuple(n,uplo,(alpha!=0)),std::make_tuple(n,uplo,(alpha!=0)),selectivex::IndexPack<0,1,2>{},
                          &cblas_dspr2,(CBLAS_ORDER)Layout,(CBLAS_UPLO)uplo,n,alpha,x,incx,y,incy,ap);
}
void selectivex_dsymv_(const int Layout, const int uplo, const int n, const double alpha, const double *a, const int lda,
            const double *x, const int incx, const double beta, double *y, const int incy){
  selectivex::selective_blas("dsymv",std::make_tuple(n,uplo,(alpha!=0)*2+(beta!=0)),
                          std::make_tuple(n,uplo,(alpha!=0)*2+(beta!=0)),selectivex::IndexPack<0,1,2>{},
                          &cblas_dsymv,(CBLAS_ORDER)Layout,(CBLAS_UPLO)uplo,n,alpha,a,lda,x,incx,beta,y,incy);
}
void selectivex_dsyr_(const int Layout, const int uplo, const int n, const double alpha, const double *x, const int incx,
            double *a, const int lda){
  selectivex::selective_blas("dsyr",std::make_tuple(n,uplo,(alpha!=0)),std::make_tuple(n,uplo,(alpha!=0)),selectivex::IndexPack<0,1,2>{},
                          &cblas_dsyr,(CBLAS_ORDER)Layout,(CBLAS_UPLO)uplo,n,alpha,x,incx,a,lda);
}
void selectivex_dsyr2_(const int Layout, const int uplo, const int n, const double alpha, const double *x, const int incx,
             const double *y, const int incy, double *a, const int lda){
  selectivex::selective_blas("dsyr2",std::make_tuple(n,uplo,(alpha!=0)),std::make_tuple(n,uplo,(alpha!=0)),selectivex::IndexPack<0,1,2>{},
                          &cblas_dsyr2,(CBLAS_ORDER)Layout,(CBLAS_UPLO)uplo,n,alpha,x,incx,y,incy,a,lda);
}
void selectivex_dtrsv_(const int order, const int uplo , const int trans , const int diag , const int n , const double *a , const int lda , double *x, const int incx ){
  selectivex::selective_blas("dtrsv",std::make_tuple(n,uplo,trans,diag),std::make_tuple(n,uplo,trans,diag),selectivex::IndexPack<0,1,2,3>{},
                          &cblas_dtrsv,(CBLAS_ORDER)order,(CBLAS_UPLO)uplo,(CBLAS_TRANSPOSE)trans,(CBLAS_DIAG)diag,n,a,lda,x,incx);
}
void selectivex_dtrmv_(const int order, const int uplo , const int trans , const int diag , const int n , const double *a , const int lda , double *x, const int incx ){
  selectivex::selective_blas("dtrmv",std::make_tuple(n,uplo,trans,diag),std::make_tuple(n,uplo,trans,diag),selectivex::IndexPack<0,1,2,3>{},
                          &cblas_dtrmv,(CBLAS_ORDER)order,(CBLAS_UPLO)uplo,(CBLAS_TRANSPOSE)trans,(CBLAS_DIAG)diag,n,a,lda,x,incx);
}
void selectivex_dtpsv_(const int order, const int uplo, const int trans, const int diag, const int n, const double *ap,
             double *x, const int incx){
  selectivex::selective_blas("dtpsv",std::make_tuple(n,uplo,trans,diag),std::make_tuple(n,uplo,trans,diag),selectivex::IndexPack<0,1,2,3>{},
                          &cblas_dtpsv,(CBLAS_ORDER)order,(CBLAS_UPLO)uplo,(CBLAS_TRANSPOSE)trans,(CBLAS_DIAG)diag,n,ap,x,incx);
}
void selectivex_dtpmv_(const int order, const int uplo, const int trans, const int diag, const int n, const double *ap,
             double *x, const int incx){
  selectivex::selective_blas("dtpmv",std::make_tuple(n,uplo,trans,diag),std::make_tuple(n,uplo,trans,diag),selectivex::IndexPack<0,1,2,3>{},
                          &cblas_dtpmv,(CBLAS_ORDER)order,(CBLAS_UPLO)uplo,(CBLAS_TRANSPOSE)trans,(CBLAS_DIAG)diag,n,ap,x,incx);
}
void selectivex_dtbsv_(const int order, const int uplo, const int trans, const int diag, const int n, const int k,
             const double *a, const int lda, double *x, const int incx){
  selectivex::selective_blas("dtbsv",std::make_tuple(n,k,uplo,trans,diag),std::make_tuple(n,k,uplo,trans,diag),selectivex::IndexPack<0,1,2,3,4>{},
                          &cblas_dtbsv,(CBLAS_ORDER)order,(CBLAS_UPLO)uplo,(CBLAS_TRANSPOSE)trans,(CBLAS_DIAG)diag,n,k,a,lda,x,incx);
}
void selectivex_dtbmv_(const int order, const int uplo, const int trans, const int diag, const int n, const int k,
             const double *a, const int lda, double *x, const int incx){
  selectivex::selective_blas("dtbmv",std::make_tuple(n,k,uplo,trans,diag),std::make_tuple(n,k,uplo,trans,diag),selectivex::IndexPack<0,1,2,3,4>{},
                          &cblas_dtbmv,(CBLAS_ORDER)order,(CBLAS_UPLO)uplo,(CBLAS_TRANSPOSE)trans,(CBLAS_DIAG)diag,n,k,a,lda,x,incx);
}

// BLAS 3
void selectivex_dgemm_(const int order, const int transa , const int transb ,
             const int m , const int n , const int k , const double alpha , const double *a ,
             const int lda , const double *b , const int ldb , const double beta , double *c , const int ldc){
  selectivex::selective_blas("dgemm",std::make_tuple(m,n,k,transa*1000+transb,(alpha!=0)*2+(beta!=0)),
                          std::make_tuple(m,n,k,transa*1000+transb,(alpha!=0)*2+(beta!=0)),selectivex::IndexPack<0,1,2,3,4>{},
                          &cblas_dgemm,(CBLAS_ORDER)order,(CBLAS_TRANSPOSE)transa,(CBLAS_TRANSPOSE)transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
void selectivex_dtrmm_(const int order, const int side , const int uplo , const int transa ,
             const int diag , const int m , const int n , const double alpha , const double *a ,
             const int lda , double *b , const int ldb){
  selectivex::selective_blas("dtrmm",std::make_tuple(m,n,side*1000+uplo,transa*1000+diag,(alpha!=0)),
                          std::make_tuple(m,n,side*1000+uplo,transa*1000+diag,(alpha!=0)),selectivex::IndexPack<0,1,2,3,4>{},
                          &cblas_dtrmm,(CBLAS_ORDER)order,(CBLAS_SIDE)side,(CBLAS_UPLO)uplo,(CBLAS_TRANSPOSE)transa,(CBLAS_DIAG)diag,m,n,alpha,a,lda,b,ldb);
}
void selectivex_dtrsm_(const int order, const int side , const int uplo , const int transa ,
             const int diag , const int m , const int n , const double alpha , const double *a ,
             const int lda , double *b , const int ldb){
  selectivex::selective_blas("dtrsm",
                          (CBLAS_SIDE)side==CblasLeft ? std::make_tuple(m,n,side*1000+uplo,transa*1000+diag,(alpha!=0)) : std::make_tuple(n,m,side*1000+uplo,transa*1000+diag,(alpha!=0)),
                          std::make_tuple(m,n,side*1000+uplo,transa*1000+diag,(alpha!=0)), selectivex::IndexPack<0,1,2,3,4>{},
                          &cblas_dtrsm,(CBLAS_ORDER)order,(CBLAS_SIDE)side,(CBLAS_UPLO)uplo,(CBLAS_TRANSPOSE)transa,(CBLAS_DIAG)diag,m,n,alpha,a,lda,b,ldb);
}
void selectivex_dsyrk_(const int order, const int uplo , const int trans ,
             const int n , const int k , const double alpha , const double *a , const int lda ,
             const double beta , double *c , const int ldc){
  selectivex::selective_blas("dsyrk",std::make_tuple(n,k,uplo*1000+trans,(alpha!=0)*2+(beta!=0)),
                          std::make_tuple(n,k,uplo*1000+trans,(alpha!=0)*2+(beta!=0)),selectivex::IndexPack<0,1,2,3>{},
                          &cblas_dsyrk,(CBLAS_ORDER)order,(CBLAS_UPLO)uplo,(CBLAS_TRANSPOSE)trans,n,k,alpha,a,lda,beta,c,ldc);
}
void selectivex_dsyr2k_(const int order, const int uplo, const int trans, const int n, const int k, const double alpha,
              const double *a, const int lda, const double *b, const int ldb, const double beta, double *c,
              const int ldc){
  selectivex::selective_blas("dsyr2k",std::make_tuple(n,k,uplo*1000+trans,(alpha!=0)*2+(beta!=0)),
                          std::make_tuple(n,k,uplo*1000+trans,(alpha!=0)*2+(beta!=0)),selectivex::IndexPack<0,1,2,3>{},
                          &cblas_dsyr2k,(CBLAS_ORDER)order,(CBLAS_UPLO)uplo,(CBLAS_TRANSPOSE)trans,n,k,alpha,a,lda,b,ldb,beta,c,ldc);
}
void selectivex_dsymm_(const int order, const int side, const int uplo, const int m, const int n, const double alpha,
             const double *a, const int lda, const double *b, const int ldb, const double beta, double *c,
             const int ldc){
  selectivex::selective_blas("dsymm",
                          (CBLAS_SIDE)side==CblasLeft ? std::make_tuple(m,n,side*1000+uplo,(alpha!=0)*2+(beta!=0)) : std::make_tuple(n,m,side*1000+uplo,(alpha!=0)*2+(beta!=0)),
                          std::make_tuple(m,n,side*1000+uplo,(alpha!=0)*2+(beta!=0)), selectivex::IndexPack<0,1,2,3>{},
                          &cblas_dsymm,(CBLAS_ORDER)order,(CBLAS_SIDE)side,(CBLAS_UPLO)uplo,m,n,alpha,a,lda,b,ldb,beta,c,ldc);
}

// **********************************************************************************************************************************
// BLAS 1
void selectivex__daxpy__(const int* n , const double* a , const double *x , const int* incx , double *y , const int* incy){
  selectivex_daxpy_(*n,*a,x,*incx,y,*incy);
}
void selectivex__dscal__(const int* n , const double* a , double *x , const int* incx){
  selectivex_dscal_(*n,*a,x,*incx);
}

// BLAS 2
void selectivex__dgbmv__(const char* trans , const int* m , const int* n, const int* kl, const int* ku, const double* alpha ,
               const double *a , const int* lda , const double *x, const int* incx ,
               const double* beta, double *y , const int* incy ){
  CBLAS_TRANSPOSE _trans;
  if (*trans=='T') _trans = CblasTrans; else if (*trans=='N') _trans = CblasNoTrans; else _trans = CblasConjTrans;
  selectivex_dgbmv_(CblasColMajor,_trans,*m,*n,*kl,*ku,*alpha,a,*lda,x,*incx,*beta,y,*incy);
}
void selectivex__dgemv__(const char* trans , const int* m , const int* n, const double* alpha , const double *a , const int* lda ,
               const double *x, const int* incx , const double* beta, double *y , const int* incy ){
  CBLAS_TRANSPOSE _trans;
  if (*trans=='T') _trans = CblasTrans; else if (*trans=='N') _trans = CblasNoTrans; else _trans = CblasConjTrans;
  selectivex_dgemv_(CblasColMajor,_trans,*m,*n,*alpha,a,*lda,x,*incx,*beta,y,*incy);
}
void selectivex__dger__(const int* m , const int* n , const double* alpha , const double *x , const int* incx , const double *y ,
              const int* incy , double *a ,
            const int* lda){
  selectivex_dger_(CblasColMajor,*m,*n,*alpha,x,*incx,y,*incy,a,*lda);
}
void selectivex__dsbmv__(const char* uplo, const int* n, const int* k, const double* alpha, const double *a,
               const int* lda, const double *x, const int* incx, const double* beta, double *y, const int* incy){
  selectivex_dsbmv_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower),*n,*k,*alpha,a,*lda,x,*incx,*beta,y,*incy);
}
void selectivex__dspmv__(const char* uplo, const int* n, const double* alpha, const double *ap, const double *x,
               const int* incx, const double* beta, double *y, const int* incy){
  selectivex_dspmv_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower),*n,*alpha,ap,x,*incx,*beta,y,*incy);
}
void selectivex__dspr__(const char* uplo, const int* n, const double* alpha, const double *x,
              const int* incx, double *ap){
  selectivex_dspr_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower),*n,*alpha,x,*incx,ap);
}
void selectivex__dspr2__(const char* uplo, const int* n, const double* alpha, const double *x, const int* incx,
               const double *y, const int* incy, double *ap){
  selectivex_dspr2_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower),*n,*alpha,x,*incx,y,*incy,ap);
}
void selectivex__dsymv__(const char* uplo, const int* n, const double* alpha, const double *a, const int* lda,
               const double *x, const int* incx, const double* beta, double *y, const int* incy){
  selectivex_dsymv_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower),*n,*alpha,a,*lda,x,*incx,*beta,y,*incy);
}
void selectivex__dsyr__(const char* uplo, const int* n, const double* alpha, const double *x, const int* incx,
              double *a, const int* lda){
  selectivex_dsyr_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower),*n,*alpha,x,*incx,a,*lda);
}
void selectivex__dsyr2__(const char* uplo, const int* n, const double* alpha, const double *x, const int* incx,
               const double *y, const int* incy, double *a, const int* lda){
  selectivex_dsyr2_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower),*n,*alpha,x,*incx,y,*incy,a,*lda);
}
void selectivex__dtrsv__(const char* uplo, const char* trans, const char* diag, const int* n, const double *a,
               const int* lda, double *x, const int* incx){
  CBLAS_TRANSPOSE _trans;
  if (*trans=='T') _trans = CblasTrans; else if (*trans=='N') _trans = CblasNoTrans; else _trans = CblasConjTrans;
  selectivex_dtrsv_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower), _trans, (*diag=='U' ? CblasUnit : CblasNonUnit), *n,a,*lda,x,*incx);
}
void selectivex__dtrmv__(const char* uplo , const char* trans , const char* diag , const int* n , const double *a ,
               const int* lda , double *x, const int* incx ){
  CBLAS_TRANSPOSE _trans;
  if (*trans=='T') _trans = CblasTrans; else if (*trans=='N') _trans = CblasNoTrans; else _trans = CblasConjTrans;
  selectivex_dtrmv_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower), _trans, (*diag=='U' ? CblasUnit : CblasNonUnit), *n,a,*lda,x,*incx);
}
void selectivex__dtpsv__(const char* uplo, const char* trans, const char* diag, const int* n, const double *ap,
               double *x, const int* incx){
  CBLAS_TRANSPOSE _trans;
  if (*trans=='T') _trans = CblasTrans; else if (*trans=='N') _trans = CblasNoTrans; else _trans = CblasConjTrans;
  selectivex_dtpsv_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower), _trans, (*diag=='U' ? CblasUnit : CblasNonUnit), *n,ap,x,*incx);
}
void selectivex__dtpmv__(const char* uplo, const char* trans, const char* diag, const int* n, const double *ap,
               double *x, const int* incx){
  CBLAS_TRANSPOSE _trans;
  if (*trans=='T') _trans = CblasTrans; else if (*trans=='N') _trans = CblasNoTrans; else _trans = CblasConjTrans;
  selectivex_dtpmv_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower), _trans, (*diag=='U' ? CblasUnit : CblasNonUnit), *n,ap,x,*incx);
}
void selectivex__dtbsv__(const char* uplo, const char* trans, const char* diag, const int* n, const int* k,
               const double *a, const int* lda, double *x, const int* incx){
  CBLAS_TRANSPOSE _trans;
  if (*trans=='T') _trans = CblasTrans; else if (*trans=='N') _trans = CblasNoTrans; else _trans = CblasConjTrans;
  selectivex_dtbsv_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower), _trans, (*diag=='U' ? CblasUnit : CblasNonUnit), *n,*k,a,*lda,x,*incx);
}
void selectivex__dtbmv__(const char* uplo, const char* trans, const char* diag, const int* n, const int* k,
               const double *a, const int* lda, double *x, const int* incx){
  CBLAS_TRANSPOSE _trans;
  if (*trans=='T') _trans = CblasTrans; else if (*trans=='N') _trans = CblasNoTrans; else _trans = CblasConjTrans;
  selectivex_dtbmv_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower), _trans, (*diag=='U' ? CblasUnit : CblasNonUnit), *n,*k,a,*lda,x,*incx);
}

// BLAS 3
void selectivex__dgemm__(const char* transa , const char* transb ,
             const int* m , const int* n , const int* k , const double* alpha , const double *a ,
             const int* lda , const double *b , const int* ldb , const double* beta , double *c , const int* ldc){
  CBLAS_TRANSPOSE _transa;
  if (*transa=='T') _transa = CblasTrans; else if (*transa=='N') _transa = CblasNoTrans; else _transa = CblasConjTrans;
  CBLAS_TRANSPOSE _transb;
  if (*transb=='T') _transb = CblasTrans; else if (*transb=='N') _transb = CblasNoTrans; else _transb = CblasConjTrans;
  selectivex_dgemm_(CblasColMajor,_transa, _transb, *m,*n,*k,*alpha,a,*lda,b,*ldb,*beta,c,*ldc);
}
void selectivex__dtrmm__(const char* side , const char* uplo , const char* transa ,
             const char* diag , const int* m , const int* n , const double* alpha , const double *a ,
             const int* lda , double *b , const int* ldb){
  CBLAS_TRANSPOSE _transa;
  if (*transa=='T') _transa = CblasTrans; else if (*transa=='N') _transa = CblasNoTrans; else _transa = CblasConjTrans;
  selectivex_dtrmm_(CblasColMajor,(*side=='L' ? CblasLeft : CblasRight), (*uplo=='U' ? CblasUpper : CblasLower), _transa,
          (*diag=='U' ? CblasUnit : CblasNonUnit), *m,*n,*alpha,a,*lda,b,*ldb);
}
void selectivex__dtrsm__(const char* side , const char* uplo , const char* transa ,
             const char* diag , const int* m , const int* n , const double* alpha , const double *a ,
             const int* lda , double *b , const int* ldb){
  CBLAS_TRANSPOSE _transa;
  if (*transa=='T') _transa = CblasTrans; else if (*transa=='N') _transa = CblasNoTrans; else _transa = CblasConjTrans;
  selectivex_dtrsm_(CblasColMajor,(*side=='L' ? CblasLeft : CblasRight), (*uplo=='U' ? CblasUpper : CblasLower), _transa,
          (*diag=='U' ? CblasUnit : CblasNonUnit), *m,*n,*alpha,a,*lda,b,*ldb);
}
void selectivex__dsyrk__(const char* uplo, const char* trans ,
             const int* n , const int* k , const double* alpha , const double *a , const int* lda ,
             const double* beta , double *c , const int* ldc){
  CBLAS_TRANSPOSE _trans;
  if (*trans=='T') _trans = CblasTrans; else if (*trans=='N') _trans = CblasNoTrans; else _trans = CblasConjTrans;
  selectivex_dsyrk_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower), _trans, *n,*k,*alpha,a,*lda,*beta,c,*ldc);
}
void selectivex__dsyr2k__(const char* uplo, const char* trans, const int* n, const int* k, const double* alpha,
                const double *a, const int* lda, const double *b, const int* ldb, const double* beta, double *c,
                const int* ldc){
  CBLAS_TRANSPOSE _trans;
  if (*trans=='T') _trans = CblasTrans; else if (*trans=='N') _trans = CblasNoTrans; else _trans = CblasConjTrans;
  selectivex_dsyr2k_(CblasColMajor,(*uplo=='U' ? CblasUpper : CblasLower), _trans,*n,*k,*alpha,a,*lda,b,*ldb,*beta,c,*ldc);
}
void selectivex__dsymm__(const char* side, const char* uplo, const int* m, const int* n, const double* alpha,
               const double *a, const int* lda, const double *b, const int* ldb, const double* beta, double *c,
               const int* ldc){
  selectivex_dsymm_(CblasColMajor,(*side=='L' ? CblasLeft : CblasRight), (*uplo=='U' ? CblasUpper : CblasLower),*m,*n,*alpha,
          a,*lda,b,*ldb,*beta,c,*ldc);
}

// **********************************************************************************************************************************
// C interface
int selectivex_dgetrf_(int matrix_layout, int m , int n , double* a , int lda , int* ipiv){
  return selectivex::selective_lapack("dgetrf",
                                 (m>=n ? std::make_tuple(m,n) : std::make_pair(n,m)), std::make_tuple(m,n),selectivex::IndexPack<0,1>{},
                                 &LAPACKE_dgetrf,std::make_tuple(matrix_layout,m,n,a,lda,ipiv),selectivex::IndexPack<0,1,2,3,4,5>{},
                                std::make_tuple(a,m,n,lda,1,[](double* a, int m, int n, int lda){for (int i=0; i<n; i++){a[i*lda+i] = 4.*n;}}));
}
int selectivex_dpotrf_(int matrix_layout, char uplo , int n , double* a , int lda){
  return selectivex::selective_lapack("dpotrf",
                                   std::make_tuple(n,uplo), std::make_tuple(n,uplo),selectivex::IndexPack<0,1>{},
                                   &LAPACKE_dpotrf,std::make_tuple(matrix_layout,uplo,n,a,lda),selectivex::IndexPack<0,1,2,3,4>{},
                                   std::make_tuple(a,n,n,lda,1,[](double* a, int m, int n, int lda){for (int i=0; i<n; i++){a[i*lda+i] = 4.*n;}}));
}
int selectivex_dtrtri_(int matrix_layout, char uplo , char diag , int n , double* a , int lda){
  return selectivex::selective_lapack("dtrtri",
                                   std::make_tuple(n,uplo,diag), std::make_tuple(n,uplo,diag),selectivex::IndexPack<0,1,2>{},
                                   &LAPACKE_dtrtri,std::make_tuple(matrix_layout,uplo,diag,n,a,lda),selectivex::IndexPack<0,1,2,3,4,5>{},
                                   std::make_tuple(a,n,n,lda,1,[](double* a, int m, int n, int lda){for (int i=0; i<n; i++){a[i*lda+i] = 4.*n;}}));
}

int selectivex_dgeqrf_(int matrix_layout, int m , int n , double* a , int lda , double* tau){
  return selectivex::selective_lapack("dgeqrf",
                                   (m>=n ? std::make_tuple(m,n) : std::make_tuple(n,m)),std::make_tuple(m,n),selectivex::IndexPack<0,1>{},
                                   &LAPACKE_dgeqrf,std::make_tuple(matrix_layout,m,n,a,lda,tau),selectivex::IndexPack<0,1,2,3,4,5>{});
                                   //std::make_tuple(a,m,n,lda,1,[](double* a, int m, int n, int lda){for (int i=0; i<n; i++){a[i*lda+i] = 4.*n;}}));
}
int selectivex_dorgqr_(int matrix_layout, int m , int n , int k , double* a , int lda , const double* tau){
  return selectivex::selective_lapack("dorgqr",
                                   std::make_tuple(m,n,k),std::make_tuple(m,n,k),selectivex::IndexPack<0,1,2>{},
                                   &LAPACKE_dorgqr,std::make_tuple(matrix_layout,m,n,k,a,lda,tau),selectivex::IndexPack<0,1,2,3,4,5,6>{});
                                   //std::make_tuple(a,m,n,lda,0,[](double* a, int m, int n, int lda){for (int i=0; i<n; i++) { a[i*lda+i] = 1.; }}));
}
int selectivex_dormqr_(int matrix_layout, char side , char trans , int m , int n , int k , const double * a , int lda , const double * tau , double * c , int ldc){
  return selectivex::selective_lapack("dormqr",
                                   (side == 'L' ? std::make_tuple(m,n,k,side,trans) : std::make_tuple(n,m,k,side,trans)),
                                   std::make_tuple(m,n,k,side,trans),selectivex::IndexPack<0,1,2,3,4>{},
                                   &LAPACKE_dormqr,std::make_tuple(matrix_layout,side,trans,m,n,k,a,lda,tau,c,ldc),selectivex::IndexPack<0,1,2,3,4,5,6,7,8,9,10>{},
                                   std::make_tuple((double*)a,(side=='L'?m:n),k,lda,0,[](double* a, int m, int n, int lda){for (int i=0; i<n; i++){a[i*lda+i] = 1.;}}));
}
int selectivex_dgetri_(int matrix_layout, int n , double * a , int lda , const int * ipiv){
  return selectivex::selective_lapack("dgetri",
                                   std::make_tuple(n),std::make_tuple(n),selectivex::IndexPack<0>{},
                                   &LAPACKE_dgetri,std::make_tuple(matrix_layout,n,a,lda,ipiv),selectivex::IndexPack<0,1,2,3,4>{},
                                   std::make_tuple(a,n,n,lda,0,[](double* a, int m, int n, int lda){for (int i=0; i<n; i++){a[i*lda+i] = 1.;}}));
}
int selectivex_dtpqrt_(int matrix_layout, int m , int n , int l , int nb , double * a , int lda , double * b , int ldb , double * t , int ldt){
  return selectivex::selective_lapack_tpqrt(
                                   &LAPACKE_dtpqrt,matrix_layout,m,n,l,nb,a,lda,b,ldb,t,ldt);
}
int selectivex_dtpmqrt_(int matrix_layout, char side , char trans , int m , int n , int k , int l , int nb , const double * v ,
               int ldv , const double * t , int ldt , double * a , int lda , double * b , int ldb){
  return selectivex::selective_lapack_tpmqrt(
                                   &LAPACKE_dtpmqrt,matrix_layout,side,trans,m,n,k,l,nb,v,ldv,t,ldt,a,lda,b,ldb);
}

// FORTRAN interface
void selectivex__dgetrf__(const int* m , const int* n , double* a , const int* lda , int* ipiv, int* info){
  *info = selectivex_dgetrf_(LAPACK_COL_MAJOR,*m,*n,a,*lda,ipiv);
}
void selectivex__dpotrf__(const char* uplo , const int* n , double* a , const int* lda, int* info){
  *info = selectivex_dpotrf_(LAPACK_COL_MAJOR,*uplo,*n,a,*lda);
}
void selectivex__dtrtri__(const char* uplo , const char* diag , const int* n , double* a , const int* lda, int* info){
  *info = selectivex_dtrtri_(LAPACK_COL_MAJOR,*uplo,*diag,*n,a,*lda);
}
void selectivex__dgeqrf__(const int* m , const int* n , double* a , const int* lda , double* tau, double* work, const int* lwork, int* info){
  *info = selectivex_dgeqrf_(LAPACK_COL_MAJOR,*m,*n,a,*lda,tau);
}
void selectivex__dorgqr__(const int* m , const int* n , const int* k , double* a , const int* lda , const double* tau, double* work, const int* lwork, int* info){
  *info = selectivex_dorgqr_(LAPACK_COL_MAJOR,*m,*n,*k,a,*lda,tau);
}
void selectivex__dormqr__(const char* side , const char* trans , const int* m , const int* n , const int* k , const double * a , const int* lda , const double * tau ,
                double * c , const int* ldc, double* work, const int* lwork, int* info){
  *info = selectivex_dormqr_(LAPACK_COL_MAJOR,*side,*trans,*m,*n,*k,a,*lda,tau,c,*ldc);
}
void selectivex__dgetri__(const int* n , double * a , const int* lda , const int * ipiv, double* work, const int* lwork, int* info){
  *info = selectivex_dgetri_(LAPACK_COL_MAJOR,*n,a,*lda,ipiv);
}
void selectivex__dtpqrt__(const int* m , const int* n , const int* l , const int* nb , double * a , const int* lda , double* b , const int* ldb , double * t , const int* ldt,
                double* work, int* info){
  *info = selectivex_dtpqrt_(LAPACK_COL_MAJOR,*m,*n,*l,*nb,a,*lda,b,*ldb,t,*ldt);
}
void selectivex__dtpmqrt__(const char* side , const char* trans , const int* m , const int* n , const int* k , const int* l , const int* nb , const double * v ,
                 const int* ldv , const double * t , const int* ldt , double * a , const int* lda , double * b , const int* ldb, double* work, int* info){
  *info = selectivex_dtpmqrt_(LAPACK_COL_MAJOR,*side,*trans,*m,*n,*k,*l,*nb,v,*ldv,t,*ldt,a,*lda,b,*ldb);
}
