#ifndef SELECTIVEX__KERNEL_REGISTRATION_HPP_
#define SELECTIVEX__KERNEL_REGISTRATION_HPP_

#include <iostream>

#include "util.h"
#include "kernel_type.h"

namespace selectivex{

template<size_t nfeatures, typename ModelType, typename... ModelArgTypes>
void register_kernel(const char* kernel_name,
                     std::function<bool(size_t*,size_t*,size_t*)> prediction_controller,
                     bool initialize_feature_models_from_file, bool initialize_model_from_file,
                     float error_tolerance, size_t min_num_samples, float min_execution_time, bool always_synchronize,
                     ModelArgTypes&&... model_args){
  // Don't register a kernel twice. Could just make this return.
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) == kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  kernel_map[kernel_name] = kernel_list.size();
  kernel_list.push_back(new internal::kernel_type<nfeatures,ModelType>(kernel_name,prediction_controller,
    initialize_feature_models_from_file,initialize_model_from_file,error_tolerance,min_num_samples,
    min_execution_time,always_synchronize,std::forward<ModelArgTypes>(model_args)...));
}

template<size_t nfeatures, typename ModelType, typename... ModelArgTypes>
void register_kernel(const char* kernel_name,
                     std::function<bool(size_t*,size_t*,size_t*)> prediction_controller,
                     bool initialize_feature_models_from_file, bool initialize_model_from_file,
                     float error_tolerance, size_t min_num_samples, float min_execution_time,
                     ModelArgTypes&&... model_args){
  // Don't register a kernel twice. Could just make this return.
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) == kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  kernel_map[kernel_name] = kernel_list.size();
  kernel_list.push_back(new internal::kernel_type<nfeatures,ModelType>(kernel_name,prediction_controller,
    initialize_feature_models_from_file,initialize_model_from_file,error_tolerance,min_num_samples,
    min_execution_time,std::forward<ModelArgTypes>(model_args)...));
}

}

#endif // SELECTIVEX__KERNEL_REGISTRATION_HPP_
