#include <iostream>

#include "interface.h"
#include "util.h"
#include "kernel_registration.h"
#include "kernel_type.h"

namespace selectivex{

#ifdef SELECTIVEX__USE_MPI
void exchange_samples(MPI_Comm cm){
  int word_size;
  MPI_Type_size(MPI_UNSIGNED_LONG, &word_size);
  assert(word_size == sizeof(size_t));
  int world_rank; MPI_Comm_rank(cm,&world_rank);
  int world_size; MPI_Comm_size(cm,&world_size);
  MPI_Status st;
  size_t active_size = world_size;
  size_t active_rank = world_rank;
  size_t active_mult = 1;
  std::vector<int> sample_envelope;
  std::vector<char> kernel_names;
  std::vector<size_t> kernel_features;
  std::vector<float> kernel_feature_stats;
  while (active_size>1){
    sample_envelope.clear();
    kernel_names.clear();
    kernel_features.clear();
    kernel_feature_stats.clear();
    if (active_rank % 2 == 1){
      int partner = (active_rank-1)*active_mult;
      for (auto& it : kernel_list){
        int local_num_observed_feature_vectors = it->get_observed_feature_vectors(kernel_features,kernel_feature_stats);
        if (local_num_observed_feature_vectors>0){
          sample_envelope.push_back(it->_kernel_name.size());
          sample_envelope.push_back(it->get_feature_vector_size());
          sample_envelope.push_back(local_num_observed_feature_vectors);
          for (int j=0; j<it->_kernel_name.size(); j++){
            kernel_names.push_back(it->_kernel_name[j]);
          }
        }  
      }
      // At this point we want to send a single message with all counts and offsets necessary to take in a message of information regarding {kernel_name,all distinct kernel_feature_vectors,{num_samples,M1,M2} for each feature vector}
      PMPI_Send(&sample_envelope[0], sample_envelope.size(), MPI_INT, partner, 66000, cm);
      PMPI_Send(&kernel_names[0], kernel_names.size(), MPI_CHAR, partner, 66001, cm);
      PMPI_Send(&kernel_features[0], kernel_features.size(), MPI_UNSIGNED_LONG, partner, 66002, cm);
      PMPI_Send(&kernel_feature_stats[0], kernel_feature_stats.size(), MPI_FLOAT, partner, 66003, cm);
      break;
    }
    else if ((active_rank % 2 == 0) && (active_rank < (active_size-1))){
      int partner = (active_rank+1)*active_mult;
      int message_size;
      MPI_Probe(partner, 66000, cm, &st);
      MPI_Get_count(&st, MPI_INT, &message_size);
      sample_envelope.resize(message_size);
      PMPI_Recv(&sample_envelope[0], message_size, MPI_INT, partner, 66000, cm, &st);
      int num_kernel_chars = 0;
      int num_kernel_features = 0;
      int num_kernel_feature_stats = 0;
      for (int j=0; j<sample_envelope.size(); j+=3){
        num_kernel_chars += sample_envelope[j];
        num_kernel_features += (sample_envelope[j+1]+1)*sample_envelope[j+2];
        num_kernel_feature_stats += sample_envelope[j+2]*2;// 2 instead of 3 because 'num_samples' is a size_t and is communicated via the kernel_feature buffer.
      }
      kernel_names.resize(num_kernel_chars);
      kernel_features.resize(num_kernel_features);
      kernel_feature_stats.resize(num_kernel_feature_stats);
      PMPI_Recv(&kernel_names[0], num_kernel_chars, MPI_CHAR, partner, 66001, cm, MPI_STATUS_IGNORE);
      PMPI_Recv(&kernel_features[0], num_kernel_features, MPI_UNSIGNED_LONG, partner, 66002, cm, MPI_STATUS_IGNORE);
      PMPI_Recv(&kernel_feature_stats[0], num_kernel_feature_stats, MPI_FLOAT, partner, 66003, cm, MPI_STATUS_IGNORE);
      int offset1=0;
      int offset2=0;
      int offset3=0;
      for (int j=0; j<sample_envelope.size(); j+=3){
        auto kernel_str = std::string(kernel_names.begin()+offset1,kernel_names.begin()+offset1+sample_envelope[j]);
        if (kernel_map.find(kernel_str) != kernel_map.end()){
          kernel_list[kernel_map[kernel_str]]->update(sample_envelope[j+2],&kernel_features[offset2],&kernel_feature_stats[offset3]);
        } else{
          //TODO: create/register new entry
          assert(0);
        }
        offset1 += sample_envelope[j];
        offset2 += (sample_envelope[j+1]+1)*sample_envelope[j+2];
        offset3 += sample_envelope[j+2]*2;// See same reason as above
      }

    } else{
      assert(world_rank != 0);// We need rank-0 to acquire all of the sample information, because only it writes to file.
    }
    active_size = active_size/2 + active_size%2;
    active_rank /= 2;
    active_mult *= 2;
  }
}
#endif // SELECTIVEX__USE_MPI

void init_variables(){
  internal::within_window = true;
  internal::within_kernel = false;
  if (std::getenv("SELECTIVEX_INITIALIZE_KERNELS") != NULL){
    internal::initialize_kernels = atoi(std::getenv("SELECTIVEX_INITIALIZE_KERNELS"))==1;
  } else{
    internal::initialize_kernels = true;
  }
  if (std::getenv("SELECTIVEX_INVOCATION_SWITCH") != NULL){
    internal::invocation_switch = atoi(std::getenv("SELECTIVEX_INVOCATION_SWITCH"))==1;
  } else{
    internal::invocation_switch = 0;
  }
  if (std::getenv("SELECTIVEX_RANDOM_EXECUTION_PERCENTAGE") != NULL){
    internal::random_execution_percentage = std::stof(std::getenv("SELECTIVEX_RANDOM_EXECUTION_PERCENTAGE"))==1;
  } else{
    internal::random_execution_percentage = 0.1;
  }
  if (std::getenv("SELECTIVEX_DATA_FOLDER") != NULL){
    internal::data_folder_path = std::string(std::getenv("SELECTIVEX_DATA_FOLDER"));
  } else{
    internal::data_folder_path = "";
  }
}

void init_kernels(){
  bool initialize_kernel_feature_data_from_file = (std::getenv("SELECTIVEX_INITIALIZE_KERNEL_FEATURE_MODELS") != NULL ? atoi(std::getenv("SELECTIVEX_INITIALIZE_KERNEL_FEATURE_MODELS"))==1 : true);
  bool initialize_kernel_models_from_file = (std::getenv("SELECTIVEX_INITIALIZE_KERNEL_MODELS") != NULL ? atoi(std::getenv("SELECTIVEX_INITIALIZE_KERNEL_MODELS"))==1 : true);
  float blas_confidence_tolerance_threshold = (std::getenv("SELECTIVEX_BLAS_CONFIDENCE_TOLERANCE") != NULL ? std::stof(std::getenv("SELECTIVEX_BLAS_CONFIDENCE_TOLERANCE")) : 0.05);
  size_t blas_min_num_executions = (std::getenv("SELECTIVEX_BLAS_MIN_NUM_EXECUTIONS") != NULL ? std::stof(std::getenv("SELECTIVEX_BLAS_MIN_NUM_EXECUTIONS")) : 3);
  float blas_min_execution_time = (std::getenv("SELECTIVEX_BLAS_MIN_EXECUTION_TIME") != NULL ? std::stof(std::getenv("SELECTIVEX_BLAS_MIN_EXECUTION_TIME")) : 1e-3);
  // Due to constraint that each kernel's number of features is a compile-time parameter,
  //   we must manually register each automatically-intercepted kernel.
#ifdef SELECTIVEX_BLAS3
  constexpr std::array<size_t,6> list_of_blas_kernel_features = {{5,5,5,4,4,4}};
  register_kernel<list_of_blas_kernel_features[0],internal::NoOpModel>(list_of_blas_kernels[0].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,blas_confidence_tolerance_threshold,blas_min_execution_time,blas_min_execution_time);
  register_kernel<list_of_blas_kernel_features[1],internal::NoOpModel>(list_of_blas_kernels[1].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,blas_confidence_tolerance_threshold,blas_min_execution_time,blas_min_execution_time);
  register_kernel<list_of_blas_kernel_features[2],internal::NoOpModel>(list_of_blas_kernels[2].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,blas_confidence_tolerance_threshold,blas_min_execution_time,blas_min_execution_time);
  register_kernel<list_of_blas_kernel_features[3],internal::NoOpModel>(list_of_blas_kernels[3].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,blas_confidence_tolerance_threshold,blas_min_execution_time,blas_min_execution_time);
  register_kernel<list_of_blas_kernel_features[4],internal::NoOpModel>(list_of_blas_kernels[4].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,blas_confidence_tolerance_threshold,blas_min_execution_time,blas_min_execution_time);
  register_kernel<list_of_blas_kernel_features[5],internal::NoOpModel>(list_of_blas_kernels[5].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,blas_confidence_tolerance_threshold,blas_min_execution_time,blas_min_execution_time);
#endif // SELECTIVEX_BLAS3

  float lapack_confidence_tolerance_threshold = (std::getenv("SELECTIVEX_LAPACK_CONFIDENCE_TOLERANCE") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_CONFIDENCE_TOLERANCE")) : 0.05);
  size_t lapack_min_num_executions = (std::getenv("SELECTIVEX_LAPACK_MIN_NUM_EXECUTIONS") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_MIN_NUM_EXECUTIONS")) : 3);
  float lapack_min_execution_time = (std::getenv("SELECTIVEX_LAPACK_MIN_EXECUTION_TIME") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_MIN_EXECUTION_TIME")) : 1e-3);
#ifdef SELECTIVEX_LAPACK
  constexpr std::array<size_t,9> list_of_lapack_kernel_features = {{2,2,3,2,3,5,1,4,5}};
  register_kernel<list_of_lapack_kernel_features[0],internal::NoOpModel>(list_of_lapack_kernels[0].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,lapack_confidence_tolerance_threshold,lapack_min_execution_time,lapack_min_execution_time);
  register_kernel<list_of_lapack_kernel_features[1],internal::NoOpModel>(list_of_lapack_kernels[1].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,lapack_confidence_tolerance_threshold,lapack_min_execution_time,lapack_min_execution_time);
  register_kernel<list_of_lapack_kernel_features[2],internal::NoOpModel>(list_of_lapack_kernels[2].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,lapack_confidence_tolerance_threshold,lapack_min_execution_time,lapack_min_execution_time);
  register_kernel<list_of_lapack_kernel_features[3],internal::NoOpModel>(list_of_lapack_kernels[3].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,lapack_confidence_tolerance_threshold,lapack_min_execution_time,lapack_min_execution_time);
  register_kernel<list_of_lapack_kernel_features[4],internal::NoOpModel>(list_of_lapack_kernels[4].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,lapack_confidence_tolerance_threshold,lapack_min_execution_time,lapack_min_execution_time);
  register_kernel<list_of_lapack_kernel_features[5],internal::NoOpModel>(list_of_lapack_kernels[5].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,lapack_confidence_tolerance_threshold,lapack_min_execution_time,lapack_min_execution_time);
  register_kernel<list_of_lapack_kernel_features[6],internal::NoOpModel>(list_of_lapack_kernels[6].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,lapack_confidence_tolerance_threshold,lapack_min_execution_time,lapack_min_execution_time);
  register_kernel<list_of_lapack_kernel_features[7],internal::NoOpModel>(list_of_lapack_kernels[7].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,lapack_confidence_tolerance_threshold,lapack_min_execution_time,lapack_min_execution_time);
  register_kernel<list_of_lapack_kernel_features[8],internal::NoOpModel>(list_of_lapack_kernels[8].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,lapack_confidence_tolerance_threshold,lapack_min_execution_time,lapack_min_execution_time);
#endif // SELECTIVEX_LAPACK

#ifdef SELECTIVEX__USE_MPI
  if (std::getenv("SELECTIVEX_P2P_PROTOCOL_SWITCH") != NULL){
    internal::p2p_protocol_switch = atoi(std::getenv("SELECTIVEX_P2P_PROTOCOL_SWITCH"));
  } else{
    internal::p2p_protocol_switch = INT_MAX;
  }
  float mpi_confidence_tolerance_threshold = (std::getenv("SELECTIVEX_MPI_CONFIDENCE_TOLERANCE") != NULL ? std::stof(std::getenv("SELECTIVEX_MPI_CONFIDENCE_TOLERANCE")) : 0.05);
  size_t mpi_min_num_executions = (std::getenv("SELECTIVEX_MPI_MIN_NUM_EXECUTIONS") != NULL ? std::stof(std::getenv("SELECTIVEX_MPI_MIN_NUM_EXECUTIONS")) : 3);
  float mpi_min_execution_time = (std::getenv("SELECTIVEX_MPI_MIN_EXECUTION_TIME") != NULL ? std::stof(std::getenv("SELECTIVEX_MPI_MIN_EXECUTION_TIME")) : 1e-3);
  bool mpi_always_synchronize = (std::getenv("SELECTIVEX_MPI_ALWAYS_SYNCHRONIZE") != NULL ? atoi(std::getenv("SELECTIVEX_MPI_ALWAYS_SYNCHRONIZE"))==1 : true);
  constexpr std::array<size_t,18> list_of_mpi_kernel_features = {{1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,4,3,3}};
  register_kernel<list_of_mpi_kernel_features[0],internal::NoOpModel>(list_of_mpi_kernels[0].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,0,mpi_min_execution_time,mpi_min_execution_time,true);
  register_kernel<list_of_mpi_kernel_features[1],internal::NoOpModel>(list_of_mpi_kernels[1].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[2],internal::NoOpModel>(list_of_mpi_kernels[2].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[3],internal::NoOpModel>(list_of_mpi_kernels[3].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[4],internal::NoOpModel>(list_of_mpi_kernels[4].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[5],internal::NoOpModel>(list_of_mpi_kernels[5].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[6],internal::NoOpModel>(list_of_mpi_kernels[6].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[7],internal::NoOpModel>(list_of_mpi_kernels[7].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[8],internal::NoOpModel>(list_of_mpi_kernels[8].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[9],internal::NoOpModel>(list_of_mpi_kernels[9].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[10],internal::NoOpModel>(list_of_mpi_kernels[10].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[11],internal::NoOpModel>(list_of_mpi_kernels[11].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[12],internal::NoOpModel>(list_of_mpi_kernels[12].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[13],internal::NoOpModel>(list_of_mpi_kernels[13].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[14],internal::NoOpModel>(list_of_mpi_kernels[14].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[15],internal::NoOpModel>(list_of_mpi_kernels[15].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,mpi_confidence_tolerance_threshold,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  // As a general rule, never skip execution of MPI_Isend/MPI_Irecv
  register_kernel<list_of_mpi_kernel_features[16],internal::NoOpModel>(list_of_mpi_kernels[16].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,0,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
  register_kernel<list_of_mpi_kernel_features[17],internal::NoOpModel>(list_of_mpi_kernels[17].c_str(),[](size_t*,size_t*,size_t*){return false;},initialize_kernel_feature_data_from_file,initialize_kernel_models_from_file,0,mpi_min_execution_time,mpi_min_execution_time,mpi_always_synchronize);
#endif // SELECTIVEX__USE_MPI
}

#ifdef SELECTIVEX__USE_MPI

void start(MPI_Comm cm, bool from_mpi_init){
#ifdef SELECTIVEX__USE_ASSERTS
  // Don't call this if you specify to initialize windows automatically
  //   upon interception of MPI_Init*
  assert(!internal::within_window);
#endif // SELECTIVEX__USE_ASSERTS
  if (std::getenv("SELECTIVEX_AUTOMATIC_START") != NULL){
    assert(from_mpi_init);// If this environment variable is set, then start should not be invoked by the user
    internal::automatic_start = true;
    internal::within_window = true;
  } else if (!from_mpi_init){
    internal::within_window = true;
  } else{
    // Getting here signifies that this routine was invoked automatically upon
    //   interception of PMPI_Init*(...), but automatic start of window is not
    //   requested.
    internal::within_window = false;// This should already be set, but this is just in case.
  }
  init_variables();

  if (internal::initialize_kernels) init_kernels();

  PMPI_Barrier(cm);
  save_accelerated_execution_clock = MPI_Wtime();
  save_reconstructed_execution_clock = MPI_Wtime();
}

void stop(MPI_Comm cm, bool from_mpi_finalize){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(internal::within_window);
#endif // SELECTIVEX__USE_ASSERTS
  double current_time = MPI_Wtime();
  accelerated_execution_time += (current_time-save_accelerated_execution_clock);
  reconstructed_execution_time += (current_time - save_reconstructed_execution_clock);
  if (internal::automatic_start){
    assert(from_mpi_finalize);// If this environment variable is set, then start should not be invoked by the user
    internal::within_window = false;
  } else if (!from_mpi_finalize){
    internal::within_window = false;
  } else{
    // Getting here signifies that this routine was invoked automatically upon
    //   interception of PMPI_Init*(...), but automatic start of window is not
    //   requested.
  }
  internal::within_kernel = false;
  std::array<double,4> local_timers = {accelerated_kernel_execution_time,reconstructed_kernel_execution_time,accelerated_execution_time,reconstructed_execution_time};
  std::array<double,4> remote_timers;
  PMPI_Allreduce(&local_timers[0],&remote_timers[0],4,MPI_DOUBLE,MPI_MAX,cm);
  accelerated_kernel_execution_time = remote_timers[0];
  reconstructed_kernel_execution_time = remote_timers[1];
  accelerated_execution_time = remote_timers[2];
  reconstructed_execution_time = remote_timers[3];

  if (internal::automatic_start){
    deregister_kernels(cm,true,true);
  }
#ifdef SELECTIVEX__PRINT_TIMERS
  int myrank;
  MPI_Comm_rank(cm,&myrank);
  if (myrank == 0){
    std::cout << "Total reconstructed kernel benchmark execution time (from selectivex) - " << selectivex::reconstructed_kernel_execution_time << std::endl;
    std::cout << "Total accelerated kernel execution time (from selectivex) - " << selectivex::accelerated_kernel_execution_time << std::endl;
    std::cout << "Total reconstructed benchmark execution time (from selectivex) - " << selectivex::reconstructed_execution_time << std::endl;
    std::cout << "Total accelerated execution time (from selectivex) - " << selectivex::accelerated_execution_time << std::endl;
  }
#endif // SELECTIVEX__PRINT_TIMERS
}

// Modeling interface for accelerated execution
// Modeling interface templated on the number of features.
void update_model(const char* kernel_name, MPI_Comm cm, bool aggregate_samples){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  kernel_list[kernel_map[kernel_name]]->update_model(cm,aggregate_samples);
/*
  .. think about when this would be invoked, and whether it will affect the global timers.
  .. ideally this is called within an autotuner, or the models are saved to file, etc.
*/
}

bool should_observe(const char* kernel_name, size_t* features, MPI_Comm cm){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  reconstructed_execution_time += (MPI_Wtime()-save_reconstructed_execution_clock);
  if (!internal::within_window) return true;
  if (internal::within_kernel) return save_kernel_execution_decision;
  bool ret = kernel_list[kernel_map[kernel_name]]->should_observe(features,cm);
  save_kernel_execution_decision = ret;
  internal::within_kernel = true;
  save_reconstructed_kernel_execution_clock = MPI_Wtime();
  return ret;
}

bool should_observe(const char* kernel_name, size_t* features, MPI_Comm cm, int partner, bool is_sender, bool is_blocking){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  reconstructed_execution_time += (MPI_Wtime()-save_reconstructed_execution_clock);
  if (!internal::within_window) return true;
  if (internal::within_kernel) return save_kernel_execution_decision;
  bool ret = kernel_list[kernel_map[kernel_name]]->should_observe(features,cm,partner,is_sender,is_blocking);
  save_kernel_execution_decision = ret;
  internal::within_kernel = true;
  save_reconstructed_kernel_execution_clock = MPI_Wtime();
  return ret;
}

bool should_observe(const char* kernel_name, size_t* features, MPI_Comm cm, int dest, int src){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  reconstructed_execution_time += (MPI_Wtime()-save_reconstructed_execution_clock);
  if (!internal::within_window) return true;
  if (internal::within_kernel) return save_kernel_execution_decision;
  bool ret = kernel_list[kernel_map[kernel_name]]->should_observe(features,cm,dest,src);
  save_kernel_execution_decision = ret;
  internal::within_kernel = true;
  save_reconstructed_kernel_execution_clock = MPI_Wtime();
  return ret;
}

void observe(const char* kernel_name, size_t* features, MPI_Comm cm, double runtime){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  if (!internal::within_window) return;
  if (save_kernel_execution_decision && runtime<0){
    // Enter here if user wants selectivex to track execution time (so default value of -1 used as runtime)
    double kernel_execution_time = MPI_Wtime() - save_reconstructed_kernel_execution_clock;
    reconstructed_kernel_execution_time += kernel_execution_time;
    reconstructed_execution_time += kernel_execution_time;
    accelerated_kernel_execution_time += kernel_execution_time;
    kernel_list[kernel_map[kernel_name]]->observe(features,kernel_execution_time,cm);
  } else if (save_kernel_execution_decision){
    // Enter here if user explicitly passes a suitable runtime parameter
    reconstructed_kernel_execution_time += runtime;
    reconstructed_execution_time += runtime;
    accelerated_kernel_execution_time += runtime;
    kernel_list[kernel_map[kernel_name]]->observe(features,runtime,cm);
  } else{
    kernel_list[kernel_map[kernel_name]]->observe(features,runtime,cm);// Basically a no-op, runtime is -1 but won't contribute
  }
  save_kernel_execution_decision = false;// reset
  internal::within_kernel = false;// reset
  save_reconstructed_execution_clock = MPI_Wtime();
}

void skip_observe(double runtime, bool is_before_wait, int* request){
  if (!internal::within_window) return;
  // Enter here if user explicitly passes a suitable runtime parameter
  if (runtime>=0) reconstructed_execution_time += runtime;
  if (is_before_wait){
    internal::save_nonblocking_info[*request] = std::make_pair(internal::nonblocking_buffer,internal::nonblocking_request);
  }
  save_kernel_execution_decision = false;// reset
  internal::within_kernel = false;// reset
  save_reconstructed_execution_clock = MPI_Wtime();
}

void skip_should_observe(int* request){
  if (!internal::within_window) return;
#ifdef SELECTIVEX__USE_ASSERTS
  assert(internal::save_nonblocking_info.find(*request) != internal::save_nonblocking_info.end());
#endif // SELECTIVEX__USE_ASSERTS
  PMPI_Wait(internal::save_nonblocking_info[*request].second,MPI_STATUS_IGNORE);
  reconstructed_kernel_execution_time = std::max(reconstructed_kernel_execution_time,internal::save_nonblocking_info[*request].first[1]);
  accelerated_kernel_execution_time = std::max(accelerated_kernel_execution_time,internal::save_nonblocking_info[*request].first[2]);
  reconstructed_execution_time = std::max(reconstructed_execution_time,internal::save_nonblocking_info[*request].first[3]);
  delete internal::save_nonblocking_info[*request].first;
  delete internal::save_nonblocking_info[*request].second;
}


void deregister_kernels(MPI_Comm cm, bool save_inputs_to_file, bool save_model_to_file){
  exchange_samples(cm);
  int mpi_rank;
  MPI_Comm_rank(cm,&mpi_rank);
  for (auto it : kernel_list){
    if (mpi_rank == 0) it->write_to_file(save_inputs_to_file,save_model_to_file);
    delete it;
  }
}

#endif /* SELECTIVEX__USE_MPI */

void start(){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(!internal::within_window);
#endif // SELECTIVEX__USE_ASSERTS
  init_variables();
  if (internal::initialize_kernels) init_kernels();
#ifdef SELECTIVEX__USE_MPI
  save_accelerated_execution_clock = MPI_Wtime();
  save_reconstructed_execution_clock = MPI_Wtime();
#else
  save_accelerated_execution_clock = std::chrono::high_resolution_clock::now();
  save_reconstructed_execution_clock = std::chrono::high_resolution_clock::now();
#endif // SELECTIVEX__USE_MPI
}

void stop(){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(internal::within_window);
#endif // SELECTIVEX__USE_ASSERTS
#ifdef SELECTIVEX__USE_MPI
  double current_time = MPI_Wtime();
  accelerated_execution_time += (current_time-save_accelerated_execution_clock);
  reconstructed_execution_time += (current_time - save_reconstructed_execution_clock);
#else
  auto save_current_clock = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> accelerated_diff = (save_current_clock-save_accelerated_execution_clock);
  accelerated_execution_time += accelerated_diff.count();
  std::chrono::duration<double> reconstructed_diff = (save_current_clock-save_reconstructed_execution_clock);
  reconstructed_execution_time += reconstructed_diff.count();
#endif
  internal::within_window = false;
}

void update_model(const char* kernel_name){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  kernel_list[kernel_map[kernel_name]]->update_model();
}

bool should_observe(const char* kernel_name, size_t* features){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
#ifdef SELECTIVEX__USE_MPI
  reconstructed_execution_time += MPI_Wtime()-save_reconstructed_execution_clock;
#else
  auto end_reconstructed_execution_clock = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_reconstructed_execution_clock - save_reconstructed_execution_clock;
  reconstructed_execution_time += diff.count();
#endif // SELECTIVEX__USE_MPI
  if (!internal::within_window) return true;
  if (internal::within_kernel) return save_kernel_execution_decision;
  bool ret = kernel_list[kernel_map[kernel_name]]->should_observe(features);
  save_kernel_execution_decision = ret;
  internal::within_kernel = true;
#ifdef SELECTIVEX__USE_MPI
  save_reconstructed_kernel_execution_clock = MPI_Wtime();
#else
  save_reconstructed_kernel_execution_clock = std::chrono::high_resolution_clock::now();
#endif // SELECTIVEX__USE_MPI
  return ret;
}

void observe(const char* kernel_name, size_t* features, double runtime){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  if (!internal::within_window) return;
  auto end_reconstructed_kernel_execution_clock = std::chrono::high_resolution_clock::now();
  if (save_kernel_execution_decision && runtime<0){
    // Enter here if user wants selectivex to track execution time (so default value of -1 used as runtime)
#ifdef SELECTIVEX__USE_MPI
    double kernel_execution_time = MPI_Wtime() - save_reconstructed_kernel_execution_clock;
    reconstructed_kernel_execution_time += kernel_execution_time;
    reconstructed_execution_time += kernel_execution_time;
    accelerated_kernel_execution_time += kernel_execution_time;
    kernel_list[kernel_map[kernel_name]]->observe(features,kernel_execution_time);
#else
    std::chrono::duration<double> diff = end_reconstructed_kernel_execution_clock - save_reconstructed_kernel_execution_clock;
    reconstructed_kernel_execution_time += diff.count();
    reconstructed_execution_time += diff.count();
    accelerated_kernel_execution_time += diff.count();
    kernel_list[kernel_map[kernel_name]]->observe(features,diff.count());
#endif
  } else if (save_kernel_execution_decision){
    // Enter here if user explicitly passes a suitable runtime parameter
    reconstructed_kernel_execution_time += runtime;
    reconstructed_execution_time += runtime;
    accelerated_kernel_execution_time += runtime;
    kernel_list[kernel_map[kernel_name]]->observe(features,runtime);
  } else{
    kernel_list[kernel_map[kernel_name]]->observe(features,runtime);// Basically a no-op, runtime is -1 but won't contribute
  }
  save_kernel_execution_decision = false;// reset
  internal::within_kernel = false;// reset
#ifdef SELECTIVEX__USE_MPI
  save_reconstructed_execution_clock = MPI_Wtime();
#else
  save_reconstructed_execution_clock = std::chrono::high_resolution_clock::now();
#endif // SELECTIVEX__USE_MPI
}

void reset_kernel(const char* kernel_name){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  kernel_list[kernel_map[kernel_name]]->reset();
}

void reset_kernel(const char* kernel_name, size_t* features){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
#endif // SELECTIVEX__USE_ASSERTS
  kernel_list[kernel_map[kernel_name]]->reset(features);
}

void deregister_kernels(bool save_inputs_to_file, bool save_model_to_file){
#ifdef SELECTIVEX__USE_MPI
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
#endif // SELECTIVEX__USE_MPI
  for (auto it : kernel_list){
#ifdef SELECTIVEX__USE_MPI
    if (mpi_rank == 0) it->write_to_file(save_inputs_to_file,save_model_to_file);
#else
    it->write_to_file(save_inputs_to_file,save_model_to_file);
#endif // SELECTIVEX__USE_MPI
    delete it;
  }
}

float predict(const char* kernel_name, size_t* features){
#ifdef SELECTIVEX__USE_ASSERTS
  assert(kernel_map.find(kernel_name) != kernel_map.end());
  assert(internal::within_window);
#endif // SELECTIVEX__USE_ASSERTS
  if (!internal::within_window) return -1;// Should never happen, see assert above
  double predicted_runtime = kernel_list[kernel_map[kernel_name]]->predict(features);
  if (!save_kernel_execution_decision){
    reconstructed_kernel_execution_time += predicted_runtime;
    reconstructed_execution_time += predicted_runtime;
  }
  return predicted_runtime;
}

void clear(const char* kernel_name){
  //TODO:What did these function calls do?
  //internal::clear_aggregates();
  //internal::generate_initial_aggregate();
  //internal::clear_counter++;
  //internal::accelerate::accelerator::clear(kernel_name);
}

void record(int variantID, int print_mode, double overhead_time){
/*
  internal::print(variantID,print_mode,overhead_time);
  internal::write_file(variantID,print_mode,overhead_time);
*/
}

void set_debug(int debug_mode){
  //NOTE: Only supposed to be used for debugging to set/save reference critical-path costs/times
/*
  internal::autotuning_debug = debug_mode;
  if (debug_mode != 0){
    if (internal::mechanism == 1){
      internal::set_reference_values();
    }
  }
  else{
     internal::save_reference_values();
  }
*/
}

}
