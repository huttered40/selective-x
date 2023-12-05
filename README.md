
# selective-x
Welcome! If you are looking for a lightweight tool to selectively execute kernels within a high-level running application, you have come to the right place. `selective-x` is a fast, header-only C++ library that **seeks to learn the execution time of individual kernels executed within data-oblivious sequential and/or distributed-memory MPI applications**.

Note that this library uses global variables and is thus not thread-safe for tracking execution time.
However, users can supply their own execution times as a work-around...

Each string must be unique corresponding to a distinct kernel being tracked.

Each kernel input is identified by a string and an array of a specified length.
Each kernel string will have its own map.

If you have a BSP-style nonblocking communication region in which many Isends/Irecvs go out, then you should explicitly invoke a custom kernel with a communicator.
Selective-x cannot handle this efficiently.

We provide both automatic control and fine-grained control.
MPI,BLAS,LAPACK routines intercepted.

If use wants BLAS/LAPACK/MPI routines to be automatically intercepted, then they need to explicitly provide the right include files and corresponding include paths and -D flags. See src/intercept/comp.cxx

If you want to use Selectivex without any intervention, then it is better to build offline and just use built library.
However, the user still has to provide a header file somewhere so that the MPI/BLAS/LAPACK routines be intercepted.
- If sequential, no MPI_Init call can be intercepted, so user must call selectivex::start and selectivex::stop to initialize a window. Else, BLAS/LAPACK/MPI kernels can be automatically intercepted, but will not get tracked.

Library should work with/without:
1. Use of critter. Should be an option
2. Automatic interception or user-controlled source code annotation to register kernels.
3. Sequential or parallel setting.
