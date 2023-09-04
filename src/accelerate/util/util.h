#ifndef SELECTIVEX__ACCELERATE__UTIL__UTIL_H_
#define SELECTIVEX__ACCELERATE__UTIL__UTIL_H_

#include "../../util/util.h"
#include "../../model/per-input.h"

namespace selectivex{
namespace internal{
namespace accelerate{

extern std::ofstream stream,stream_kernel,stream_tune,stream_reconstruct;
extern bool schedule_decision;
extern int measure_kernel_time,reset_kernel_execution_state,reset_kernel_distribution;
extern int propagate_kernel_execution_state;
extern int kernel_execution_count_mode,schedule_kernels,update_analysis;
extern int kernel_execution_control_mode,debug_iter_count;
extern int delay_state_update,collective_state_protocol;
extern float min_kernel_execution_time,error_tolerance,min_kernel_execution_percentage;
extern float* save_path_data;
extern MPI_Request save_prop_req;
extern volatile double comp_start_time;
extern size_t min_num_kernel_execution_count,mode_1_width,mode_2_width;
extern size_t max_track_kernel_count;
extern size_t num_cp_measures,num_tracker_cp_measures;
extern size_t cp_costs_size,aux_info_size;
extern int internal_tag,internal_tag1,internal_tag2,internal_tag3,internal_tag4,internal_tag5;
extern bool is_first_iter;
extern std::map<int,kernel> model_container;
extern std::vector<kernel_per_input> models;
// ****************************************************************************************************************************************************
//extern MPI_Datatype kernel_type,batch_type;
//extern std::map<comm_kernel_key,kernel> comm_kernel_save_map;
//extern std::map<comp_kernel_key,kernel> comp_kernel_save_map;
//extern std::map<comm_kernel_key,kernel> comm_kernel_ref_map;
//extern std::map<comp_kernel_key,kernel> comp_kernel_ref_map;
//extern std::map<comm_kernel_key,std::vector<kernel_batch>> comm_batch_map;
//extern std::map<comp_kernel_key,std::vector<kernel_batch>> comp_batch_map;
//extern std::map<comm_kernel_key,std::vector<kernel>> comm_kernel_list;
//extern std::map<comp_kernel_key,std::vector<kernel>> comp_kernel_list;
// ****************************************************************************************************************************************************
extern float intercept_overhead;
extern float global_intercept_overhead;
extern std::vector<float> global_kernel_stats;
extern std::vector<float> local_kernel_stats;
extern std::vector<float> save_kernel_stats;
extern std::vector<float> cp_costs;
extern std::vector<float> cp_costs_foreign;
extern std::vector<float> cp_costs_ref;
extern std::vector<char> eager_pad;
extern std::map<MPI_Request,nonblocking_info> nonblocking_internal_info;

void initialize(MPI_Comm comm);
void final_accumulate(MPI_Comm comm, double last_time);
void reset();
void clear(const std::string& kernel_name);
void reference_initiate();
void reference_transfer();
void finalize();

}
}
}

#endif /*SELECTIVEX__ACCELERATE__UTIL__UTIL_H_*/
