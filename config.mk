CXX=mpicc
#DEFS=-DUSE_SELECTIVEX -DSELECTIVEX_BLAS3 -DSELECTIVEX__USE_CBLAS_INTERFACE_INDI -DSELECTIVEX_LAPACK -DSELECTIVEX__USE_MPI -DSELECTIVEX_MPI_COLLECTIVES -DSELECTIVEX_MPI_P2P -DSELECTIVEX__PRINT_TIMERS
DEFS=-DUSE_SELECTIVEX -DSELECTIVEX__USE_CBLAS_INTERFACE -DSELECTIVEX_BLAS3 -DSELECTIVEX_LAPACK -DSELECTIVEX__USE_MPI -DSELECTIVEX_MPI_COLLECTIVES -DSELECTIVEX_MPI_P2P -DSELECTIVEX__PRINT_TIMERS
INCLUDES=
CXXFLAGS=-g -O3 $(DEFS) -std=c++0x -fPIC $(INCLUDES)
LDFLAGS= 