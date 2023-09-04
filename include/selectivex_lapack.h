#ifndef __SELECTIVEX__LAPACK_H__
#define __SELECTIVEX__LAPACK_H__

#include "../src/interface.h"
#include "../src/intercept/comp.h"

// C interface
#define LAPACKE_dgetrf(a,b,c,d,e,f)\
    selectivex_dgetrf_(a,b,c,d,e,f)

#define LAPACKE_dpotrf(a,b,c,d,e)\
    selectivex_dpotrf_(a,b,c,d,e)

#define LAPACKE_dtrtri(a,b,c,d,e,f)\
    selectivex_dtrtri_(a,b,c,d,e,f)

#define LAPACKE_dgeqrf(a,b,c,d,e,f)\
    selectivex_dgeqrf_(a,b,c,d,e,f)

#define LAPACKE_dorgqr(a,b,c,d,e,f,g)\
    selectivex_dorgqr_(a,b,c,d,e,f,g)

#define LAPACKE_dormqr(a,b,c,d,e,f,g,h,i,j,k)\
    selectivex_dormqr_(a,b,c,d,e,f,g,h,i,j,k)

#define LAPACKE_dgetri(a,b,c,d,e)\
    selectivex_dgetri_(a,b,c,d,e)

#define LAPACKE_dtpmqrt(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)\
    selectivex_dtpmqrt_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)

#define LAPACKE_dtpqrt(a,b,c,d,e,f,g,h,i,j,k)\
    selectivex_dtpqrt_(a,b,c,d,e,f,g,h,i,j,k)


// FORTRAN interface
#define dgetrf_(a,b,c,d,e,f)\
    selectivex__dgetrf__(a,b,c,d,e,f)
#define dgetrf(a,b,c,d,e,f)\
    selectivex__dgetrf__(a,b,c,d,e,f)
#define DGETRF_(a,b,c,d,e,f)\
    selectivex__dgetrf__(a,b,c,d,e,f)
#define DGETRF(a,b,c,d,e,f)\
    selectivex__dgetrf__(a,b,c,d,e,f)

#define dpotrf_(a,b,c,d,e)\
    selectivex__dpotrf__(a,b,c,d,e)
#define dpotrf(a,b,c,d,e)\
    selectivex__dpotrf__(a,b,c,d,e)
#define DPOTRF_(a,b,c,d,e)\
    selectivex__dpotrf__(a,b,c,d,e)
#define DPOTRF(a,b,c,d,e)\
    selectivex__dpotrf__(a,b,c,d,e)

#define dtrtri_(a,b,c,d,e,f)\
    selectivex__dtrtri__(a,b,c,d,e,f)
#define dtrtri(a,b,c,d,e,f)\
    selectivex__dtrtri__(a,b,c,d,e,f)
#define DTRTRI_(a,b,c,d,e,f)\
    selectivex__dtrtri__(a,b,c,d,e,f)
#define DTRTRI(a,b,c,d,e,f)\
    selectivex__dtrtri__(a,b,c,d,e,f)

#define dgeqrf_(a,b,c,d,e,f,g,h)\
    selectivex__dgeqrf__(a,b,c,d,e,f,g,h)
#define dgeqrf(a,b,c,d,e,f,g,h)\
    selectivex__dgeqrf__(a,b,c,d,e,f,g,h)
#define DGEQRF_(a,b,c,d,e,f,g,h)\
    selectivex__dgeqrf__(a,b,c,d,e,f,g,h)
#define DGEQRF(a,b,c,d,e,f,g,h)\
    selectivex__dgeqrf__(a,b,c,d,e,f,g,h)

#define dorgqr_(a,b,c,d,e,f,g,h,i)\
    selectivex__dorgqr__(a,b,c,d,e,f,g,h,i)
#define dorgqr(a,b,c,d,e,f,g,h,i)\
    selectivex__dorgqr__(a,b,c,d,e,f,g,h,i)
#define DORGQR_(a,b,c,d,e,f,g,h,i)\
    selectivex__dorgqr__(a,b,c,d,e,f,g,h,i)
#define DORGQR(a,b,c,d,e,f,g,h,i)\
    selectivex__dorgqr__(a,b,c,d,e,f,g,h,i)

#define dormqr_(a,b,c,d,e,f,g,h,i,j,k,l,m)\
    selectivex__dormqr__(a,b,c,d,e,f,g,h,i,j,k,l,m)
#define dormqr(a,b,c,d,e,f,g,h,i,j,k,l,m)\
    selectivex__dormqr__(a,b,c,d,e,f,g,h,i,j,k,l,m)
#define DORMQR_(a,b,c,d,e,f,g,h,i,j,k,l,m)\
    selectivex__dormqr__(a,b,c,d,e,f,g,h,i,j,k,l,m)
#define DORMQR(a,b,c,d,e,f,g,h,i,j,k,l,m)\
    selectivex__dormqr__(a,b,c,d,e,f,g,h,i,j,k,l,m)

#define dgetri_(a,b,c,d,e,f,g)\
    selectivex__dgetri__(a,b,c,d,e,f,g)
#define dgetri(a,b,c,d,e,f,g)\
    selectivex__dgetri__(a,b,c,d,e,f,g)
#define DGETRI_(a,b,c,d,e,f,g)\
    selectivex__dgetri__(a,b,c,d,e,f,g)
#define DGETRI(a,b,c,d,e,f,g)\
    selectivex__dgetri__(a,b,c,d,e,f,g)

#define dtpmqrt_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)\
    selectivex__dtpmqrt__(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)
#define dtpmqrt(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)\
    selectivex__dtpmqrt__(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)
#define DTPMQRT_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)\
    selectivex__dtpmqrt__(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)
#define DTPMQRT(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)\
    selectivex__dtpmqrt__(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)

#define dtpqrt_(a,b,c,d,e,f,g,h,i,j,k,l)\
    selectivex__dtpqrt__(a,b,c,d,e,f,g,h,i,j,k,l)
#define dtpqrt(a,b,c,d,e,f,g,h,i,j,k,l)\
    selectivex__dtpqrt__(a,b,c,d,e,f,g,h,i,j,k,l)
#define DTPQRT_(a,b,c,d,e,f,g,h,i,j,k,l)\
    selectivex__dtpqrt__(a,b,c,d,e,f,g,h,i,j,k,l)
#define DTPQRT(a,b,c,d,e,f,g,h,i,j,k,l)\
    selectivex__dtpqrt__(a,b,c,d,e,f,g,h,i,j,k,l)

#endif // __SELECTIVEX__LAPACK_H__
