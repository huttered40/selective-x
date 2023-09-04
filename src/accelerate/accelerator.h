#ifndef SELECTIVEX__ACCELERATE__ACCELERATOR_H_
#define SELECTIVEX__ACCELERATE__ACCELERATOR_H_

#include "../container/comm_tracker.h"

namespace selectivex{
namespace internal{
namespace accelerate{

class accelerator{
public:
  static void exchange_communicators(MPI_Comm oldcomm, MPI_Comm newcomm);
  template <size_t nfeatures>
  static bool should_observe(std:string kernel_name, volatile double curtime, int nfeatures, float* features, MPI_Comm cm=MPI_COMM_NULL);
  template <size_t nfeatures>
  void observe(std::string kernel_name, volatile double kernel_time, float* features, MPI_Comm cm=MPI_COMM_NULL);

  static bool initiate_comm(blocking& tracker, volatile double curtime, int64_t nelem, MPI_Datatype t, MPI_Comm comm,
                       bool is_sender, int partner1, int user_tag1, int partner2, int user_tag2);
/*
  static bool initiate_comm(nonblocking& tracker, volatile double curtime, int64_t nelem,
                       MPI_Datatype t, MPI_Comm comm, bool is_sender, int partner, int user_tag);
  static void initiate_comm(nonblocking& tracker, volatile double itime, int64_t nelem,
                       MPI_Datatype t, MPI_Comm comm, MPI_Request* request, bool is_sender, int partner, int user_tag);
  static void complete_comm(blocking& tracker);
  static int complete_comm(double curtime, MPI_Request* request, MPI_Status* status);
  static int complete_comm(double curtime, int count, MPI_Request array_of_requests[], int* indx, MPI_Status* status);
  static int complete_comm(double curtime, int incount, MPI_Request array_of_requests[], int* outcount, int array_of_indices[], MPI_Status array_of_statuses[]);
  static int complete_comm(double curtime, int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);

  static void update_model(const char* kernel_name, int* tensor_dimension_lengths, int* interpolation_modes, int* spacing,
                    int cp_rank, float reg, int response_transform, int nals_sweeps);
  static float predict(const char* kernel_name, int nfeatures, float* features);
  static void observe(const char* kernel_name, float runtime, int nfeatures, float* features, MPI_Comm cm);
  static bool should_observe(const char* kernel_name, int nfeatures, float* features, MPI_Comm cm);
*/
private:
  static void complete_comm(nonblocking& tracker,float* path_data,  MPI_Request* request, double comp_time, double comm_time);
};

}
}
}

#endif /*SELECTIVEX__ACCELERATE__ACCELERATOR_H_*/
