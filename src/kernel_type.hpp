#ifndef SELECTIVEX__KERNEL_TYPE_HPP_
#define SELECTIVEX__KERNEL_TYPE_HPP_

#include "util.h"

namespace selectivex{

namespace internal{

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::record_feature(size_t* features){
  for (size_t i=0; i<nfeatures; i++){
    this->temporary_feature[i] = features[i];
  }
}

template<size_t nfeatures, typename ModelType>
template<typename... ModelArgTypes>
kernel_type<nfeatures,ModelType>::kernel_type(std::string kernel_name,
                            std::function<bool(size_t*,size_t*,size_t*)> prediction_controller,
                            bool initialize_feature_models_from_file, bool initialize_model_from_file,
                            float error_tolerance, size_t min_num_samples, float min_execution_time,
                            ModelArgTypes&&... model_args){
  this->feature_has_been_observed = false;
  this->_error_tolerance = error_tolerance;
  this->_min_num_samples = min_num_samples;
  this->_min_execution_time = min_execution_time;
  this->_prediction_controller = prediction_controller;
  this->_always_synchronize = false;
  this->_kernel_name = kernel_name;
  this->model_controller = Model<ModelType>(std::forward<ModelArgTypes>(model_args)...);
  if (initialize_feature_models_from_file){
    std::string read_str = this->_kernel_name + "_feature_models.csv";
    std::ifstream read_file;
    read_file.open(read_str,std::ios_base::in);
    std::array<size_t,nfeatures> feature_array;
    std::string line,_kernel_name_,temp_kernel_str,temp_feature;
    std::getline(read_file,line);// Read in header
    while(std::getline(read_file,line)){
      std::istringstream iss(line);
      std::getline(iss,temp_kernel_str,',');	// Read in string, but we don't need this
      assert(temp_kernel_str == this->_kernel_name);
      size_t count = 0;
      for (size_t count=0; count<nfeatures; count++){
        getline(iss,temp_feature,',');
        feature_array[count] = stof(temp_feature);
      }
      this->record_feature(&feature_array[0]);
      this->register_feature();
      this->recorded_features_list[this->recorded_features[this->temporary_feature]].read_from_file(iss);
    }
  }
  if (initialize_model_from_file){
    // TODO
  }
}

template<size_t nfeatures, typename ModelType>
int kernel_type<nfeatures,ModelType>::diagnose_prediction_setting(){
  if (this->feature_has_been_observed){
    return 0;
  } else{
    return 1;
  }
  return 0;
}

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::register_feature(size_t _num_samples, float _M1, float _M2, bool _is_steady, bool _is_active){
  // A feature need only be registered once
  if (this->recorded_features.find(this->temporary_feature) != this->recorded_features.end()){
    this->feature_has_been_observed = true;
    return;
  }
  this->recorded_features[this->temporary_feature] = this->recorded_features_list.size();
  this->recorded_features_list.push_back(kernel_feature_model(_num_samples,_M1,_M2,_is_steady,_is_active));
  for (size_t feature_id = 0; feature_id<nfeatures; feature_id++){
    this->feature_min[feature_id] = std::min(this->temporary_feature[feature_id],this->feature_min[feature_id]);
    this->feature_max[feature_id] = std::max(this->temporary_feature[feature_id],this->feature_max[feature_id]);
  }
  this->feature_has_been_observed = true;
}

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::write_features_to_file(){
  if (this->recorded_features_list.size()<=0) return;
  std::string write_str = data_folder_path + "/" + this->_kernel_name + "_feature_models.csv";
  std::ofstream write_file;
  write_file.open(write_str,std::ios_base::out);// Don't append
  size_t count = 0;
  for (auto& it : this->recorded_features){
    if (count==0) this->recorded_features_list[it.second].write_header_to_file(write_file,nfeatures);
    write_file << this->_kernel_name;
    for (size_t i=0; i<nfeatures; i++){
      write_file << "," << it.first[i];
    }
    this->recorded_features_list[it.second].write_to_file(write_file);
    count++;
  }
  write_file.close();
}

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::update_model(){
  this->write_features_to_file();
  this->model_controller.train();
}

template<size_t nfeatures, typename ModelType>
bool kernel_type<nfeatures,ModelType>::should_observe(size_t* features){
  this->record_feature(features);
  this->register_feature();
  if (this->diagnose_prediction_setting() == 0){
    if (this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active){
      return true;
    } else{
      return ((float) rand()/RAND_MAX) <= random_execution_percentage;
    }
  } else{
    //TODO: Should also take into account stats of other kernels, but this definitely would be outside of user control
    //      We just want the user to tell us whether or not to consider interpolation or extrapolation based on a distance from min/max
    //      on features that the user deems relevant.
    return this->_prediction_controller(&this->feature_min[0],&this->feature_max[0],&this->temporary_feature[0]);
  }
}

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::observe(size_t* features, double runtime){
  this->recorded_features_list[this->recorded_features[this->temporary_feature]].update_model(runtime);
  this->set_execution_time_steady(this->recorded_features_list[this->recorded_features[this->temporary_feature]]);
  this->set_execution_time_active(this->recorded_features_list[this->recorded_features[this->temporary_feature]]);
  this->feature_has_been_observed = false;
}

#ifdef SELECTIVEX__USE_MPI

template<size_t nfeatures, typename ModelType>
template<typename... ModelArgTypes>
kernel_type<nfeatures,ModelType>::kernel_type(std::string kernel_name,
                            std::function<bool(size_t*,size_t*,size_t*)> prediction_controller,
                            bool initialize_feature_models_from_file, bool initialize_model_from_file,
                            float error_tolerance, size_t min_num_samples, float min_execution_time, bool always_synchronize,
                            ModelArgTypes&&... model_args) :
  kernel_type(kernel_name,prediction_controller,initialize_feature_models_from_file,initialize_model_from_file,
              error_tolerance,min_num_samples,min_execution_time,std::forward<ModelArgTypes>(model_args)...){
  this->_always_synchronize = always_synchronize;
}

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::update_model(MPI_Comm cm, bool aggregate_samples){
  assert(!aggregate_samples);
  //TODO: If we aggregate samples, we must communicate.
  this->write_features_to_file();
  this->model_controller.train();
}

template<size_t nfeatures, typename ModelType>
bool kernel_type<nfeatures,ModelType>::should_observe(size_t* features, MPI_Comm cm){
  this->record_feature(features);
  this->register_feature();
  if (this->_always_synchronize || this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active){
    // Whether a kernel is active or not must be achieved via synchronization
    int rval = (((float) rand()/RAND_MAX) <= random_execution_percentage)==true;
    local_transfer_buffer[0] = (this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_steady ? rval : 1);
    local_transfer_buffer[1] = reconstructed_kernel_execution_time;
    local_transfer_buffer[2] = accelerated_kernel_execution_time;
    local_transfer_buffer[3] = reconstructed_execution_time;
    PMPI_Allreduce(&local_transfer_buffer[0],&remote_transfer_buffer[0],4,MPI_DOUBLE,MPI_MAX,cm);
    this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active = (remote_transfer_buffer[0]==1);
    reconstructed_kernel_execution_time = remote_transfer_buffer[1];
    accelerated_kernel_execution_time = remote_transfer_buffer[2];
    reconstructed_execution_time = remote_transfer_buffer[3];
  }
  if (this->diagnose_prediction_setting() == 0){
    return this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active;
  } else{
    //TODO: Should also take into account stats of other kernels, but this definitely would be outside of user control
    //      We just want the user to tell us whether or not to consider interpolation or extrapolation based on a distance from min/max
    //      on features that the user deems relevant.
    assert(0);// not clear what to do here, but some sort of synchronization would be necessary
    return this->_prediction_controller(&this->feature_min[0],&this->feature_max[0],&this->temporary_feature[0]);
  }
}

template<size_t nfeatures, typename ModelType>
bool kernel_type<nfeatures,ModelType>::should_observe(size_t* features, MPI_Comm cm, int partner, bool is_sender, bool is_blocking){
  this->record_feature(features);
  this->register_feature();
  if (is_blocking){
    if (this->_always_synchronize || this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active){
      // Whether a kernel is active or not must be achieved via synchronization
      int rval = (((float) rand()/RAND_MAX) <= random_execution_percentage)==true;
      local_transfer_buffer[0] = (this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_steady ? rval : 1);
      local_transfer_buffer[1] = reconstructed_kernel_execution_time;
      local_transfer_buffer[2] = accelerated_kernel_execution_time;
      local_transfer_buffer[3] = reconstructed_execution_time;
      if (temporary_feature[0] >= p2p_protocol_switch){
        PMPI_Sendrecv(&local_transfer_buffer[0],4,MPI_DOUBLE,partner,65000,&remote_transfer_buffer[0],4,MPI_DOUBLE,partner,65000,cm,MPI_STATUS_IGNORE);
        this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active = (remote_transfer_buffer[0]==1 && local_transfer_buffer[0]==1);
        reconstructed_kernel_execution_time = std::max(local_transfer_buffer[1],remote_transfer_buffer[1]);
        accelerated_kernel_execution_time = std::max(local_transfer_buffer[2],remote_transfer_buffer[2]);
        reconstructed_execution_time = std::max(local_transfer_buffer[3],remote_transfer_buffer[3]);
      } else{
        if (is_sender){
          PMPI_Send(&local_transfer_buffer[0],4,MPI_DOUBLE,partner,65000,cm);
          this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active = (local_transfer_buffer[0]==1);
        } else{
          PMPI_Recv(&remote_transfer_buffer[0],4,MPI_DOUBLE,partner,65000,cm,MPI_STATUS_IGNORE);
          this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active = (remote_transfer_buffer[0]==1);
          reconstructed_kernel_execution_time = std::max(local_transfer_buffer[1],remote_transfer_buffer[1]);
          accelerated_kernel_execution_time = std::max(local_transfer_buffer[2],remote_transfer_buffer[2]);
          reconstructed_execution_time = std::max(local_transfer_buffer[3],remote_transfer_buffer[3]);
        }
      }
    }
  } else{
    if (this->_always_synchronize || this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active){
      // Whether a kernel is active or not must be achieved via synchronization
      int rval = (((float) rand()/RAND_MAX) <= random_execution_percentage)==true;
      nonblocking_buffer = (double*)malloc(4*sizeof(double));
      nonblocking_request = (int*)malloc(sizeof(int)); 
      nonblocking_buffer[0] = (this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_steady ? rval : 1);
      nonblocking_buffer[1] = reconstructed_kernel_execution_time;
      nonblocking_buffer[2] = accelerated_kernel_execution_time;
      nonblocking_buffer[3] = reconstructed_execution_time;
      if (is_sender){
        PMPI_Isend(&nonblocking_buffer[0],4,MPI_DOUBLE,partner,65000,cm,nonblocking_request);
      } else{
        PMPI_Irecv(&nonblocking_buffer[0],4,MPI_DOUBLE,partner,65000,cm,nonblocking_request);
      }
    }
  }
  bool ret;
  if (this->diagnose_prediction_setting() == 0){
    ret = this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active;
  } else{
    //TODO: Should also take into account stats of other kernels, but this definitely would be outside of user control
    //      We just want the user to tell us whether or not to consider interpolation or extrapolation based on a distance from min/max
    //      on features that the user deems relevant.
    assert(0);// not clear what to do here, but some sort of synchronization would be necessary
    ret = this->_prediction_controller(&this->feature_min[0],&this->feature_max[0],&this->temporary_feature[0]);
  }
  this->feature_has_been_observed = false;
  return ret;
}

template<size_t nfeatures, typename ModelType>
bool kernel_type<nfeatures,ModelType>::should_observe(size_t* features, MPI_Comm cm, int dest, int src){
  this->record_feature(features);
  this->register_feature();
  if (this->_always_synchronize || this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active){
    // Whether a kernel is active or not must be achieved via synchronization
    int rval = (((float) rand()/RAND_MAX) <= random_execution_percentage)==true;
    local_transfer_buffer[0] = (this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_steady ? rval : 1);
    local_transfer_buffer[1] = reconstructed_kernel_execution_time;
    local_transfer_buffer[2] = accelerated_kernel_execution_time;
    local_transfer_buffer[3] = reconstructed_execution_time;
    //TODO: Should be a switch based on size of message (first feature), or simply from intercept/comm.cxx, because rendezvous protocol is needed for larger messages, thus necessitating a synchronization on the sender side (which is not needed for eager protocol).
    PMPI_Sendrecv(&local_transfer_buffer[0],4,MPI_DOUBLE,dest,65000,&remote_transfer_buffer[0],4,MPI_DOUBLE,src,65000,cm,MPI_STATUS_IGNORE);
    this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active = (remote_transfer_buffer[0]==1 && local_transfer_buffer[0]==1);
    reconstructed_kernel_execution_time = std::max(local_transfer_buffer[1],remote_transfer_buffer[1]);
    accelerated_kernel_execution_time = std::max(local_transfer_buffer[2],remote_transfer_buffer[2]);
    reconstructed_execution_time = std::max(local_transfer_buffer[3],remote_transfer_buffer[3]);
  }
  if (this->diagnose_prediction_setting() == 0){
    return this->recorded_features_list[this->recorded_features[this->temporary_feature]].is_active;
  } else{
    //TODO: Should also take into account stats of other kernels, but this definitely would be outside of user control
    //      We just want the user to tell us whether or not to consider interpolation or extrapolation based on a distance from min/max
    //      on features that the user deems relevant.
    assert(0);// not clear what to do here, but some sort of synchronization would be necessary
    return this->_prediction_controller(&this->feature_min[0],&this->feature_max[0],&this->temporary_feature[0]);
  }
}

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::observe(size_t* features, double runtime, MPI_Comm cm){
  this->recorded_features_list[this->recorded_features[this->temporary_feature]].update_model(runtime);
  // Steadiness can be achieved without communication
  this->set_execution_time_steady(this->recorded_features_list[this->recorded_features[this->temporary_feature]]);
  this->feature_has_been_observed = false;
}

#endif // SELECTIVEX__USE_MPI

template<size_t nfeatures, typename ModelType>
float kernel_type<nfeatures,ModelType>::predict(size_t* features){
  if (this->feature_has_been_observed){
    return this->recorded_features_list[this->recorded_features[this->temporary_feature]].get_estimate();
  } else{
    return this->model_controller.predict(this->temporary_feature);
  } return -1;
}

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::write_to_file(bool save_inputs_to_file, bool save_model_to_file){
  if (save_inputs_to_file){
    this->write_features_to_file();
  }
  if (save_model_to_file){
    // TODO: Not implemented yet.
/*
    std::string write_str = this->_kernel_name + "_model.csv";
    std::ofstream write_file;
    write_file.open(write_str,std::ios_base::out);// Don't append
    this->model_controller.write_to_file(write_file,this->_kernel_name);
    write_file.close();
*/
  }
}

template<size_t nfeatures, typename ModelType>
int kernel_type<nfeatures,ModelType>::get_observed_feature_vectors(std::vector<size_t>& feature_vectors, std::vector<float>& feature_stats){
  int count = 0;
  for (auto& it : this->recorded_features){
    if (this->recorded_features_list[it.second].num_samples>0){
      count++;
      for (int i=0; i<nfeatures; i++){
        feature_vectors.push_back(it.first[i]);
      }
      feature_vectors.push_back(this->recorded_features_list[it.second].num_samples);
      feature_stats.push_back(this->recorded_features_list[it.second].M1);
      feature_stats.push_back(this->recorded_features_list[it.second].M2);
    }
  }
  return count;
}

template<size_t nfeatures, typename ModelType>
int kernel_type<nfeatures,ModelType>::get_feature_vector_size(){
  return nfeatures;
}

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::update(int num_features, size_t* feature_pointer, float* stats_pointer){
  for (int i=0; i<num_features; i++){
    this->record_feature(feature_pointer);
    if (this->recorded_features.find(this->temporary_feature) != this->recorded_features.end()){
      this->recorded_features_list[this->recorded_features[this->temporary_feature]].update((size_t)stats_pointer[0],stats_pointer[1],stats_pointer[2]); 
    } else{
      this->register_feature((size_t)stats_pointer[0],stats_pointer[1],stats_pointer[2]);
    }
  }
}

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::reset(){
  for (auto& it : this->recorded_features_list){
    it.clear_model();
  }
}

template<size_t nfeatures, typename ModelType>
void kernel_type<nfeatures,ModelType>::reset(size_t* features){
  this->record_feature(features);
  assert(this->recorded_features.find(this->temporary_feature) != this->recorded_features.end());
  this->recorded_features_list[this->recorded_features[this->temporary_feature]].clear_model();
}

}

}

#endif // SELECTIVEX__KERNEL_TYPE_HPP_
