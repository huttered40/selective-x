#ifndef SELECTIVEX__KERNEL_TYPE_H_
#define SELECTIVEX__KERNEL_TYPE_H_

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

#include "model.h"
#include "util.h"

namespace selectivex{
namespace internal{

template<size_t nfeatures, typename ModelType>
class kernel_type : public abstract_kernel_type {
public:
  template<typename... ModelArgTypes>
  kernel_type(std::string kernel_name,
              std::function<bool(size_t*,size_t*,size_t*)> prediction_controller,
              bool initialize_feature_models_from_file, bool initialize_model_from_file,
              float error_tolerance, size_t min_num_samples, float min_execution_time,
              ModelArgTypes&&... model_args);
  template<typename... ModelArgTypes>
  kernel_type(std::string kernel_name,
              std::function<bool(size_t*,size_t*,size_t*)> prediction_controller,
              bool initialize_feature_models_from_file, bool initialize_model_from_file,
              float error_tolerance, size_t min_num_samples, float min_execution_time, bool always_synchronize,
              ModelArgTypes&&... model_args);
  int diagnose_prediction_setting();
  void update_model() override;
  bool should_observe(size_t* features) override;
  void observe(size_t* features, double runtime) override;
#ifdef SELECTIVEX__USE_MPI 
  void update_model(MPI_Comm cm, bool aggregate_samples) override;
  bool should_observe(size_t* features, MPI_Comm cm) override;
  bool should_observe(size_t* features, MPI_Comm cm, int partner, bool is_sender, bool is_blocking) override;
  bool should_observe(size_t* features, MPI_Comm c, int dest, int srcm) override;
  void observe(size_t* features, double runtime, MPI_Comm cm) override;
#endif // SELECTIVEX__USE_MPI
  float predict(size_t* features) override;
  void write_to_file(bool save_inputs_to_file, bool save_model_to_file) override;
  void reset() override;
  void reset(size_t* features) override;
  int get_observed_feature_vectors(std::vector<size_t>& feature_vectors, std::vector<float>& feature_stats) override;
  int get_feature_vector_size() override;
  void update(int num_features, size_t* feature_pointer, float* stats_pointer) override;

private:
  inline void record_feature(size_t* features);
  inline void register_feature(size_t _num_samples=0, float _M1=0, float _M2=0, bool _is_steady=false, bool _is_active=true);
  void write_features_to_file();

  Model<ModelType> model_controller;

  std::map<std::array<size_t,nfeatures>,size_t> recorded_features;
  std::vector<kernel_feature_model> recorded_features_list;

  std::array<size_t,nfeatures> temporary_feature;
  std::array<size_t,nfeatures> feature_min;
  std::array<size_t,nfeatures> feature_max;
};

}
}

#include "kernel_type.hpp"

#endif /*SELECTIVEX__KERNEL_TYPE_H_*/
