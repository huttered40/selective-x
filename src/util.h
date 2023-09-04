#ifndef SELECTIVEX__UTIL_H_
#define SELECTIVEX__UTIL_H_

#ifdef SELECTIVEX__USE_MPI
#include <mpi.h>
#else
#endif /* SELECTIVEX__USE_MPI */

#include <cassert>
#include <map>
#include <cmath>
#include <functional>
#include <vector>
#include <assert.h>
#include <climits>

#include <sstream>

namespace selectivex{
namespace internal{

class kernel_feature_model{
public:
  kernel_feature_model(size_t _num_samples, float _M1, float _M2, bool _is_steady, bool _is_active);
  kernel_feature_model(const kernel_feature_model& copy);
  void update_model(double runtime);
  void clear_model();
  void update(size_t remote_num_samples, float remote_M1, float remote_M2);
  void write_header_to_file(std::ofstream& write_file, size_t nfeatures);
  void write_to_file(std::ofstream& write_file);
  void read_from_file(std::istringstream& iss);
  float get_estimate();
  float get_variance();
  float get_standard_deviation();

  bool is_steady;
  bool is_active;
  size_t num_samples;
  size_t num_invocations;
  float M1;
  float M2;
/*
  int num_schedules;
  int hash_id;
  std::set<channel*> registered_channels;
  std::list<model*> parents;
*/
};

class abstract_kernel_type{
public:
  virtual void update_model() = 0;
  virtual bool should_observe(size_t* features) = 0;
  virtual void observe(size_t* features, double runtime) = 0;
#ifdef SELECTIVEX__USE_MPI 
  virtual void update_model(MPI_Comm cm, bool aggregate_samples) = 0;
  virtual bool should_observe(size_t* features, MPI_Comm cm) = 0;
  virtual bool should_observe(size_t* features, MPI_Comm cm, int partner, bool is_sender, bool is_blocking) = 0;
  virtual bool should_observe(size_t* features, MPI_Comm cm, int dest, int src) = 0;
  virtual void observe(size_t* features, double runtime, MPI_Comm cm) = 0;
#endif // SELECTIVEX__USE_MPI
  virtual float predict(size_t* features) = 0;
  virtual void write_to_file(bool save_inputs_to_file, bool save_model_to_file) = 0;
  virtual void reset() = 0;
  virtual void reset(size_t* features) = 0;
  virtual int get_observed_feature_vectors(std::vector<size_t>& feature_vectors, std::vector<float>& feature_stats) = 0;
  virtual int get_feature_vector_size() = 0;
  virtual void update(int num_features, size_t* feature_pointer, float* stats_pointer) = 0;
protected:
  void set_execution_time_active(kernel_feature_model& m);
  void set_execution_time_steady(kernel_feature_model& m);
  float get_confidence_interval(kernel_feature_model& m);
  size_t get_count_for_confidence_internal(kernel_feature_model& m);
public:
  bool feature_has_been_observed;
  bool _always_synchronize;
  float _error_tolerance;
  size_t _min_num_samples;
  float _min_execution_time;
  std::string _kernel_name;
  std::function<bool(size_t*,size_t*,size_t*)> _prediction_controller;
};

class NoOpModel{
public:
  NoOpModel();
  float Train();
  float Predict();
};

}
}

namespace selectivex{

extern bool save_kernel_execution_decision;
extern double reconstructed_kernel_execution_time;
extern double accelerated_kernel_execution_time;
extern double reconstructed_execution_time;
extern double accelerated_execution_time;
extern int request_id;

#ifdef SELECTIVEX__USE_MPI
extern double save_accelerated_kernel_execution_clock;
extern double save_reconstructed_kernel_execution_clock;
extern double save_accelerated_execution_clock;
extern double save_reconstructed_execution_clock;
#else
extern std::chrono::time_point<std::chrono::high_resolution_clock> save_accelerated_kernel_execution_clock;
extern std::chrono::time_point<std::chrono::high_resolution_clock> save_reconstructed_kernel_execution_clock;
extern std::chrono::time_point<std::chrono::high_resolution_clock> save_accelerated_execution_clock;
extern std::chrono::time_point<std::chrono::high_resolution_clock> save_reconstructed_execution_clock;
#endif // SELECTIVEX__USE_MPI

// List of automatically-intercepted BLAS/LAPACK/MPI kernels
extern std::array<std::string,6> list_of_blas_kernels;
extern std::array<std::string,9> list_of_lapack_kernels;
extern std::array<std::string,18> list_of_mpi_kernels;

namespace internal{

#ifdef SELECTIVEX__USE_MPI
extern std::array<double,4> local_transfer_buffer;
extern std::array<double,4> remote_transfer_buffer;
extern double* nonblocking_buffer;
extern int* nonblocking_request;
extern std::map<int,std::pair<double*,int*>> save_nonblocking_info;
#endif // SELECTIVEX__USE_MPI

extern std::string data_folder_path;
extern bool within_window;
extern bool within_kernel;
extern bool automatic_start;
extern bool initialize_kernels;
extern size_t p2p_protocol_switch;
extern size_t invocation_switch;
extern float random_execution_percentage;

}

extern std::map<std::string,size_t> kernel_map;
extern std::vector<internal::abstract_kernel_type*> kernel_list;

}


#endif /*SELECTIVEX__UTIL_H_*/
