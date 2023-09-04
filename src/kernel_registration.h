#ifndef SELECTIVEX__KERNEL_REGISTRATION_H_
#define SELECTIVEX__KERNEL_REGISTRATION_H_

#include <functional>

namespace selectivex{

/* register_kernel: registers individual kernel */
template<size_t nfeatures, typename ModelType, typename... ModelArgTypes>
void register_kernel(const char* kernel_name,
                     std::function<bool(size_t*,size_t*,size_t*)> prediction_controller,
                     bool initialize_feature_models_from_file, bool initialize_model_from_file,
                     float error_tolerance, size_t min_num_samples, float min_execution_time, bool always_synchronize,
                     ModelArgTypes&&... model_args);
template<size_t nfeatures, typename ModelType, typename... ModelArgTypes>
void register_kernel(const char* kernel_name,
                     std::function<bool(size_t*,size_t*,size_t*)> prediction_controller,
                     bool initialize_feature_models_from_file, bool initialize_model_from_file,
                     float error_tolerance, size_t min_num_samples, float min_execution_time,
                     ModelArgTypes&&... model_args);
}

#include "kernel_registration.hpp"

#endif /*SELECTIVEX__KERNEL_REGISTRATION_H_*/
