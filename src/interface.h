#ifndef SELECTIVEX__INTERFACE_H_
#define SELECTIVEX__INTERFACE_H_

#ifdef SELECTIVEX__USE_MPI
#include <mpi.h>
#endif // SELECTIVEX__USE_MPI
#include <functional>
#include <chrono>

namespace selectivex{

// General purpose routines
#ifdef SELECTIVEX__USE_MPI

/* start: establishes the start of a window within which selectivex operates */
void start(MPI_Comm cm, bool from_mpi_init=false);
/* stop: establishes the end of a window within which selectivex operates */
void stop(MPI_Comm cm, bool from_mpi_finalize=false);

// Modeling interface for accelerated execution
// Modeling interface templated on the number of features.
/* update_model: update kernel-wise multi-parameter performance model (across all registered inputs) */
void update_model(const char* kernel_name, MPI_Comm cm, bool aggregate_samples = false);
/* observe: registered a kernel input and its associated measured execution time */
void observe(const char* kernel_name, size_t* features, MPI_Comm cm, double runtime = -1);
/* should_observe: check whether a kernel invocation can be skipped */
bool should_observe(const char* kernel_name, size_t* features, MPI_Comm cm);
/* should_observe: check whether a kernel invocation can be skipped */
bool should_observe(const char* kernel_name, size_t* features, MPI_Comm cm, int partner, bool is_sender, bool is_blocking=true);
/* should_observe: check whether a kernel invocation can be skipped */
bool should_observe(const char* kernel_name, size_t* features, MPI_Comm cm, int dest, int src);
/* skip_observe: restart the timer and do not time anything */
void skip_observe(double runtime = -1, bool is_before_wait = false, int* request = nullptr);
/* skip_should_observe: restart the timer and do not time anything */
void skip_should_observe(int* request = nullptr);
/* deregister_kernels: unregisters all kernels */
void deregister_kernels(MPI_Comm cm, bool save_inputs_to_file = true, bool save_model_to_file = true);

#endif /* SELECTIVEX__USE_MPI */

/* start: establishes the start of a window within which selectivex operates */
void start();
/* stop: establishes the end of a window within which selectivex operates */
void stop();

// Modeling interface for accelerated execution
// Modeling interface templated on the number of features.
/* update_model: update kernel-wise multi-parameter performance model (across all registered inputs) */
void update_model(const char* kernel_name);
/* observe: registered a kernel input and its associated measured execution time */
void observe(const char* kernel_name, size_t* features, double runtime = -1);
/* should_observe: check whether a kernel invocation can be skipped */
bool should_observe(const char* kernel_name, size_t* features);

/* reset_kernel: resets statistics tracked for all individual kernels */
void reset_kernel(const char* kernel_name);
/* reset_kernel: resets statistics tracked for all individual kernels */
void reset_kernel(const char* kernel_name, size_t* features);
/* deregister_kernels: unregisters all kernels */
void deregister_kernels(bool save_inputs_to_file = true, bool save_model_to_file = true);
/* clear: removes saved kernel information */
void clear(const char* kernel_name);
/* predict: predict execution time update kernel-wise performance model (across all registered inputs) */
float predict(const char* kernel_name, size_t* features);

// Auxiliary routines
/* set_debug: resets the mechanism (possibly different than environment variable SELECTIVEX_MECHANISM */
void set_debug(int debug_mode);
/* record: prints statistics dependent on mechanism */
void record(int variantID=-1, int print_mode=1, float overhead_time=0.);

}

#endif /*SELECTIVEX__INTERFACE_H_*/
