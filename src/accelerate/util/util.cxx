#include <limits.h>

#include "util.h"
#include "../../skeletonize/util/util.h"
#include "../../profile/util/util.h"
#include "../container/comm_tracker.h"

namespace selectivex{
namespace internal{
namespace accelerate{

std::ofstream stream,stream_kernel,stream_tune,stream_reconstruct;
bool schedule_decision;
int measure_kernel_time,reset_kernel_distribution,reset_kernel_execution_state;
int kernel_execution_count_mode,schedule_kernels,update_analysis;
int kernel_execution_control_mode,debug_iter_count;
int delay_state_update,collective_state_protocol;
int propagate_kernel_execution_state;
float min_kernel_execution_time,error_tolerance,min_kernel_execution_percentage;
size_t max_track_kernel_count;
float* save_path_data;
MPI_Request save_prop_req;
volatile double comp_start_time;
size_t min_num_kernel_execution_count,mode_1_width,mode_2_width;
size_t num_cp_measures,num_tracker_cp_measures;
size_t cp_costs_size,aux_info_size;
int internal_tag,internal_tag1,internal_tag2,internal_tag3,internal_tag4,internal_tag5;
bool is_first_iter;
std::map<std::string,kernel_info> model_container;
std::vector<kernel_per_input> models;
// ****************************************************************************************************************************************************
//MPI_Datatype kernel_type,batch_type;
//std::map<comm_kernel_key,kernel> comm_kernel_save_map;
//std::map<comp_kernel_key,kernel> comp_kernel_save_map;
//std::map<comm_kernel_key,kernel> comm_kernel_ref_map;
//std::map<comp_kernel_key,kernel> comp_kernel_ref_map;
//std::map<comm_kernel_key,std::vector<kernel_batch>> comm_batch_map;
//std::map<comp_kernel_key,std::vector<kernel_batch>> comp_batch_map;
//std::map<comm_kernel_key,std::vector<kernel>> comm_kernel_list;
//std::map<comp_kernel_key,std::vector<kernel>> comp_kernel_list;
// ****************************************************************************************************************************************************
float intercept_overhead;
float global_intercept_overhead;
std::vector<float> global_kernel_stats;
std::vector<float> local_kernel_stats;
std::vector<float> save_kernel_stats;
std::vector<float> cp_costs;
std::vector<float> cp_costs_foreign;
std::vector<float> cp_costs_ref;
std::vector<char> eager_pad;
std::map<MPI_Request,nonblocking_info> nonblocking_internal_info;


void initialize(MPI_Comm comm){
  mode_1_width = 35; mode_2_width = 15;
  communicator_count=INT_MIN;// to avoid conflict with p2p, which could range from (-p,p)
  internal_tag = 31133; internal_tag1 = internal_tag+1;
  internal_tag2 = internal_tag+2; internal_tag3 = internal_tag+3;
  internal_tag4 = internal_tag+4; internal_tag5 = internal_tag+5;
  is_first_iter = true;
  intercept_overhead = 0;
  global_intercept_overhead = 0;
  global_comm_kernel_stats.resize(5,0);
  global_comp_kernel_stats.resize(5,0);
  local_comm_kernel_stats.resize(5,0);
  local_comp_kernel_stats.resize(5,0);
  save_comp_kernel_stats.resize(2,0);
  save_comm_kernel_stats.resize(2,0);

  // Reset these global variables, as some are updated by function arguments for convenience
  if (std::getenv("SELECTIVEX_VIZ_FILE") != NULL){
    std::string stream_name = std::getenv("SELECTIVEX_VIZ_FILE");
    std::string stream_tune_name = std::getenv("SELECTIVEX_VIZ_FILE");
    std::string stream_reconstruct_name = std::getenv("SELECTIVEX_VIZ_FILE");
    std::string stream_kernel_name = std::getenv("SELECTIVEX_VIZ_FILE");
    stream_name += "_accelerate.txt";
    stream_tune_name += "_accelerate_tune.txt";
    stream_reconstruct_name += "_accelerate_reconstruct.txt";
    stream_kernel_name += "_commk.txt";
    if (is_world_root){
      stream.open(stream_name.c_str(),std::ofstream::app);
      stream_tune.open(stream_tune_name.c_str(),std::ofstream::app);
      stream_reconstruct.open(stream_reconstruct_name.c_str(),std::ofstream::app);
      stream_kernel.open(stream_kernel_name.c_str(),std::ofstream::app);
    }
  }
  // measure_kernel_time==1 signifies that selectivex measures only kernel execution times along the critical path (else it measures total execution time)
  if (std::getenv("SELECTIVEX_MEASURE_KERNEL_TIME") != NULL){
    measure_kernel_time = atoi(std::getenv("SELECTIVEX_MEASURE_KERNEL_TIME"));
    assert(measure_kernel_time>=0 && measure_kernel_time<=1);
  } else{
    measure_kernel_time = 1;
  }
  // reset_kernel_distribution: invoked when user calls selectivex::clear(...). Resets state and distribution of each kernel.
  if (std::getenv("SELECTIVEX_RESET_KERNEL_DISTRIBUTION") != NULL){
    reset_kernel_distribution = atoi(std::getenv("SELECTIVEX_RESET_KERNEL_DISTRIBUTION"));
    assert(reset_kernel_distribution >=0 && reset_kernel_distribution <=1);
  } else{
    reset_kernel_distribution = 1;
  }
  // reset_kernel_execution_state: relevant upon invocation of 'reset' of each kernel. If '1',
  //   it will ensure each kernel's 'state' is on (i.e., kernel will be executed at least once
  //   regardless of its M1/M2 statistics.
  if (std::getenv("SELECTIVEX_RESET_KERNEL_STATE") != NULL){
    reset_kernel_execution_state = atoi(std::getenv("SELECTIVEX_RESET_KERNEL_STATE"));
    assert(reset_kernel_execution_state >=0 && reset_kernel_execution_state <=1);
  } else{
    reset_kernel_execution_state = 1;
  }
  // min_num_kernel_execution_count: minimum number of kernel executions before referring to kernel's
  //   execution statistics to classify as "steady" (i.e., not necessary to be executed).
  if (std::getenv("SELECTIVEX_MIN_NUM_KERNEL_EXECUTION_COUNT") != NULL){
    min_num_kernel_execution_count = atoi(std::getenv("SELECTIVEX_MIN_NUM_KERNEL_EXECUTION_COUNT"));
    assert(min_num_kernel_execution_count >= 0);
  } else{
    min_num_kernel_execution_count = 2;
  }
  // min_kernel_execution_time: minimum execution time spent relative to each kernel before referring to kernel's
  //   execution statistics to classify as "steady" (i.e., not necessary to be executed).
  if (std::getenv("SELECTIVEX_MIN_KERNEL_EXECUTION_TIME") != NULL){
    min_kernel_execution_time = atof(std::getenv("SELECTIVEX_MIN_KERNEL_EXECUTION_TIME"));
    assert(min_kernel_execution_time >= 0.);
  } else{
    min_kernel_execution_time = 0;
  }
  // min_kernel_execution_percentage: minimum percentage of a kernel's total execution time before referring to kernel's
  //   execution statistics to classify as "steady" (i.e., not necessary to be executed).
  if (std::getenv("SELECTIVEX_MIN_KERNEL_EXECUTION_PERCENTAGE") != NULL){
    min_kernel_execution_percentage = atof(std::getenv("SELECTIVEX_MIN_KERNEL_EXECUTION_PERCENTAGE"));
    assert(min_kernel_execution_percentage >= 0.);
  } else{
    min_kernel_execution_percentage = .001;
  }
  // maximum prediction error (at the per-configuration level), translated down to standard error in kernel execution
  //   prediction.
  if (std::getenv("SELECTIVEX_ERROR_TOLERANCE") != NULL){
    error_tolerance = atof(std::getenv("SELECTIVEX_ERROR_TOLERANCE"));
    assert(error_tolerance >= 0.);
  } else{
    error_tolerance = 0;
  }
  // propagate_kernel_execution_state: controls whether a kernel's execution state is propagated about a suitable communicator
  //                         0: off
  //                         1: TODO
  //                         2: TODO
  if (std::getenv("SELECTIVEX_PROPAGATE_KERNEL_EXECUTION_STATE") != NULL){
    propagate_kernel_execution_state = atoi(std::getenv("SELECTIVEX_PROPAGATE_KERNEL_EXECUTION_STATE"));
    assert(propagate_kernel_execution_state>=0 && propagate_kernel_execution_state<=2);
  } else{
    propagate_kernel_execution_state = 0;
  }
  // kernel_execution_count_mode: used to control the kernel execution count necessary to calculate standard error
  //		             '-1': fix kernel count to be 1 regardless of number of executions (kernel never executed, used as debug case).	
  //                         '1': kernel counts along each process's execution path (no communication of CP costs needed).
  //                         '0': fix kernel count to be 1 regardless of number of executions
  //                         '2' seems to be sub-critical-path kernel counts detected online
  //                         '3': use (cost-based) critical path to determine kernel execution count
  if (std::getenv("SELECTIVEX_KERNEL_EXECUTION_COUNT_MODE") != NULL){
    kernel_execution_count_mode = atoi(std::getenv("SELECTIVEX_KERNEL_EXECUTION_COUNT_MODE"));
    assert(kernel_execution_count_mode>=-1 && kernel_execution_count_mode<=3);
  } else{
    kernel_execution_count_mode = 0;
  }
  //kernel_execution_control_mode: specifies how to control kernel execution
  //      0: percentage-based criterion (for debugging)
  //      1: confidence interval length
  //      2: always execute and save the kernel (for debugging)
  if (std::getenv("SELECTIVEX_KERNEL_EXECUTION_CONTROL_MODE") != NULL){
    kernel_execution_control_mode = atof(std::getenv("SELECTIVEX_KERNEL_EXECUTION_CONTROL_MODE"));
    assert(kernel_execution_control_mode >=0 && kernel_execution_control_mode <= 2);
  } else{
    kernel_execution_control_mode = 1;// confidence interval length
  }
  assert(kernel_execution_control_mode != 2);//TODO: This is used to debug kernels, controls comm/comp_kernel_map
  // delay_state_update: relevant for nonblocking (especially p2p) communication
  // TODO: what else ...
  if (std::getenv("SELECTIVEX_DELAY_STATE_UPDATE") != NULL){
    delay_state_update = atof(std::getenv("SELECTIVEX_DELAY_STATE_UPDATE"));
    assert(delay_state_update >= 0 && delay_state_update <= 1);
  } else{
    delay_state_update = 0;
  }
  // collective_state_protocol: specifies whether ALL processes involved in a collective communication
  //   should be steady in order to forgo execution, or just one of them.
  if (std::getenv("SELECTIVEX_COLLECTIVE_STATE_PROTOCOL") != NULL){
    collective_state_protocol = atof(std::getenv("SELECTIVEX_COLLECTIVE_STATE_PROTOCOL"));
    assert(collective_state_protocol >= 0 && collective_state_protocol <= 1);
  } else{
    collective_state_protocol = 1;
  }
  if (std::getenv("SELECTIVEX_MAX_TRACK_KERNEL_COUNT") != NULL){
    max_track_kernel_count = atof(std::getenv("SELECTIVEX_MAX_TRACK_KERNEL_COUNT"));
    assert(max_track_kernel_count>=0);
  } else{
    max_track_kernel_count = 0;
  }

  debug_iter_count = 1;
  // If user wants percentage-based stopping criterion, force knowledge of kernel frequency via skeletonize
  if (kernel_execution_control_mode==0) assert(kernel_execution_count_mode == 3);

  // Total execution time or kernel execution time (dependent on environment variable)
  num_cp_measures = 1;

  // 8 key members in 'comp_kernel' and 8 key members in 'comm_kernel'
  // +1 for each is the schedule count itself
  //TODO: 5 is a magic number. See path.cxx for reason why.
  cp_costs_size = num_cp_measures + 5;
  aux_info_size = cp_costs_size;// used to avoid magic numbers in other files
  if (kernel_execution_count_mode == 2) cp_costs_size += 9*max_track_kernel_count;

  cp_costs.resize(cp_costs_size);
  cp_costs_foreign.resize(cp_costs_size);
  cp_costs_ref.resize(cp_costs_size);

  int eager_msg_size;
  MPI_Pack_size(cp_costs_size,MPI_FLOAT,comm,&eager_msg_size);
  int eager_pad_size = MPI_BSEND_OVERHEAD;
  eager_pad_size += eager_msg_size;
  eager_pad.resize(eager_pad_size);
}

void final_accumulate(MPI_Comm comm, double last_time){
  assert(nonblocking_internal_info.size() == 0);
  cp_costs[0]+=(last_time-computation_timer);	// update critical path runtime

  _wall_time = wall_timer[wall_timer.size()-1];

/* TODO: Fix this below when I replace comm/comp and unify the kernels. Also redo the indexing
  float temp_costs[1+3+5+5+2+3+3];
  for (int i=0; i<18; i++){ temp_costs[4+i]=0; }

  accelerate::_MPI_Barrier.comm = MPI_COMM_WORLD;
  accelerate::_MPI_Barrier.partner1 = -1;
  accelerate::_MPI_Barrier.partner2 = -1;
  accelerate::_MPI_Barrier.save_comp_key.clear();
  accelerate::_MPI_Barrier.save_comm_key.clear();
  for (auto& it : model_container){
    if ((active_kernels[it.second.val_index].is_steady==1) && (should_schedule(it.second)==1)){
      accelerate::_MPI_Barrier.save_comp_key.push_back(it.first);
      temp_costs[num_cp_measures+13]++;
    }

    temp_costs[num_cp_measures+3] += active_kernels[it.second.val_index].num_local_schedules;
    temp_costs[num_cp_measures+5] += active_kernels[it.second.val_index].num_local_scheduled_units;
    temp_costs[num_cp_measures+7] += active_kernels[it.second.val_index].total_local_exec_time;
    temp_costs[num_cp_measures+15] += active_kernels[it.second.val_index].num_local_schedules;
    temp_costs[num_cp_measures+16] += active_kernels[it.second.val_index].num_local_scheduled_units;
    temp_costs[num_cp_measures+17] += active_kernels[it.second.val_index].total_local_exec_time;
  }

  for (auto i=0; i<num_cp_measures; i++) temp_costs[i] = cp_costs[i];
  temp_costs[num_cp_measures] = intercept_overhead[0];
  temp_costs[num_cp_measures+1] = intercept_overhead[1];
  temp_costs[num_cp_measures+2] = intercept_overhead[2];
  PMPI_Allreduce(MPI_IN_PLACE,&temp_costs[0],1+3+5+5+2+3+3,MPI_FLOAT,MPI_MAX,comm);
  for (auto i=0; i<num_cp_measures; i++) cp_costs[i] = temp_costs[i];
  if (autotuning_debug == 0){
    global_intercept_overhead[0] += temp_costs[num_cp_measures];
    global_intercept_overhead[1] += temp_costs[num_cp_measures+1];
    global_intercept_overhead[2] += temp_costs[num_cp_measures+2];
    global_comp_kernel_stats[0] += temp_costs[num_cp_measures+3];
    global_comp_kernel_stats[1] = temp_costs[num_cp_measures+4];
    global_comp_kernel_stats[2] += temp_costs[num_cp_measures+5];
    global_comp_kernel_stats[3] = temp_costs[num_cp_measures+6];
    global_comp_kernel_stats[4] += temp_costs[num_cp_measures+7];
    global_comm_kernel_stats[0] += temp_costs[num_cp_measures+8];
    global_comm_kernel_stats[1] = temp_costs[num_cp_measures+9];
    global_comm_kernel_stats[2] += temp_costs[num_cp_measures+10];
    global_comm_kernel_stats[3] = temp_costs[num_cp_measures+11];
    global_comm_kernel_stats[4] += temp_costs[num_cp_measures+12];
    local_comp_kernel_stats[0] = temp_costs[num_cp_measures+15];
    local_comp_kernel_stats[1] = global_comp_kernel_stats[1] - save_comp_kernel_stats[0];
    local_comp_kernel_stats[2] = temp_costs[num_cp_measures+16];
    local_comp_kernel_stats[3] = global_comp_kernel_stats[3] - save_comp_kernel_stats[1];
    local_comp_kernel_stats[4] = temp_costs[num_cp_measures+17];
    local_comm_kernel_stats[0] = temp_costs[num_cp_measures+18];
    local_comm_kernel_stats[1] = global_comm_kernel_stats[1] - save_comm_kernel_stats[0];
    local_comm_kernel_stats[2] = temp_costs[num_cp_measures+19];
    local_comm_kernel_stats[3] = global_comm_kernel_stats[3] - save_comm_kernel_stats[1];
    local_comm_kernel_stats[4] = temp_costs[num_cp_measures+20];
    save_comp_kernel_stats[0] = global_comp_kernel_stats[1];
    save_comp_kernel_stats[1] = global_comp_kernel_stats[3];
    save_comm_kernel_stats[0] = global_comm_kernel_stats[1];
    save_comm_kernel_stats[1] = global_comm_kernel_stats[3];
  }
  accelerate::_MPI_Barrier.aggregate_comp_kernels = temp_costs[num_cp_measures+13]>0;
  accelerate::_MPI_Barrier.aggregate_comm_kernels = temp_costs[num_cp_measures+14]>0;
  accelerate::_MPI_Barrier.should_propagate = accelerate::_MPI_Barrier.aggregate_comp_kernels>0 || accelerate::_MPI_Barrier.aggregate_comm_kernels>0;
*/
  // Just don't propagate the final kernels.
}

// Reset per-configuration statistics (e.g., critical-path costs/time).
void reset(){
  assert(nonblocking_internal_info.size() == 0);
  memset(&cp_costs[0],0,sizeof(float)*cp_costs.size());
  memset(&cp_costs_foreign[0],0,sizeof(float)*cp_costs.size());
  bsp_counter=0;
  intercept_overhead=0;
  memset(&local_comp_kernel_stats[0],0,sizeof(float)*local_comp_kernel_stats.size());
  memset(&local_comm_kernel_stats[0],0,sizeof(float)*local_comm_kernel_stats.size());

  // This reset will no longer reset the kernel state, but will reset the schedule counters
  for (auto& it : model_container){
    reset_kernel(active_kernels[it.second.val_index]);
    if (reset_kernel_execution_state){
      it.second.is_active = false;
      active_kernels[it.second.val_index].is_steady= false;
      active_kernels[it.second.val_index].is_active = false;
    }
  }

  if (std::getenv("SELECTIVEX_MODE") != NULL){
    mode = atoi(std::getenv("SELECTIVEX_MODE"));
    assert(mode >=0 && mode <=1);
  } else{
    mode = 1;
  }
  if (std::getenv("SELECTIVEX_SCHEDULE_KERNELS") != NULL){
    schedule_kernels = atoi(std::getenv("SELECTIVEX_SCHEDULE_KERNELS"));
    assert(schedule_kernels >=0 && schedule_kernels <=1);
  } else{
    schedule_kernels = 1;
  }
  // Note: schedule_kernels_override was default true before if (schedule_kernels==1){ schedule_kernels = (schedule_kernels_override ? schedule_kernels : 0); }
  update_analysis = 1;
/* Note: force_steady_statistical_data_overide was default false before
  if (force_steady_statistical_data_overide){
    // This branch is to be entered only after tuning a space of algorithmic parameterizations, in which the expectation is that all kernels,
    //   both comm and comp, have reached a sufficiently-predictable state (steady state).
    for (auto it : model_container){
      set_kernel_state(it.second,false);
      set_kernel_state_global(it.second,false);
    }
    update_analysis=0;
  }
*/
}

// Invoked with a list of kernels to reset. Note that if we require the user to specify a maximum history size,
//   then we can do without this. It's ad-hoc. Example is recursive Cholesky factorization with changing base-case environments,
//     but similar kernel inputs.
// Clear is a much bigger deal than Reset.
void clear(const std::string& kernel_name){

  if (reset_kernel_distribution==1){
    // I don't see any reason to clear the communicator map. In fact, doing so would be harmful
    // Actually, the batch_maps will be empty anyways, as per the loops in 'final_accumulate'.
    // The kernel keys don't really need to be updated/cleared if the active/steady buffer logic isn't being used
    //   (which it isn't), so in fact the only relevant part of this routine is the clearing of the pathsets if necessary.
    for (auto& it : model_container){
      it.second.clear_distribution();
    }
/*
    for (auto& it : kernel_ref_container){
      clear_kernel_distribution(it.second);
    }
*/
  }
  // Else clear the distributions of specific kernels passed in a list.
  else{
    for (auto& it : model_container){
      if (it.first.tag == distribution_tags[i]){
        it.second.clear_distribution();
        break;
      }
    }
/*
    for (auto& it : kernel_ref_container){
      for (int i=0; i<tag_count; i++){
        if (it.first.tag == distribution_tags[i]){
          clear_kernel_distribution(it.second);
          break;
        }
      }
    }
*/
  }
}

/* TODO: Fix this later if necessary
void reference_initiate(){
  error_tolerance=0;
  kernel_execution_control_mode = 1;
  min_kernel_execution_percentage=1;
  // Save the existing distributions so that the reference variant doesn't erase or infect them
  // Reset the distributions so that the reference doesn't take the autotuned distribution from previous configurations
  //   if reset_kernel_distribution == 0
  for (auto& it : model_container){
    comp_kernel_save_map[it.first] = active_kernels[it.second.val_index];
    if (comp_kernel_ref_map.find(it.first) != comp_kernel_ref_map.end()){
      active_kernels[it.second.val_index] = comp_kernel_ref_map[it.first];
    } else{
      reset_kernel(active_kernels[it.second.val_index]);
      if (reset_kernel_execution_state){
        active_kernels[it.second.val_index].is_steady = false;
        active_kernels[it.second.val_index].is_active = false;
      }
      clear_kernel_distribution(active_kernels[it.second.val_index]);
    }
  }
}
*/

/* TODO: Fix this later if necessary
void reference_transfer(){
  if (std::getenv("SELECTIVEX_ERROR_TOLERANCE") != NULL){ error_tolerance = atof(std::getenv("SELECTIVEX_ERROR_TOLERANCE")); }
  if (std::getenv("SELECTIVEX_MIN_KERNEL_EXECUTION_PERCENTAGE_LIMIT") != NULL){ min_kernel_execution_percentage = atof(std::getenv("SELECTIVEX_MIN_KERNEL_EXECUTION_PERCENTAGE")); }
  if (std::getenv("SELECTIVEX_KERNEL_EXECUTION_CONTROL_MODE") != NULL){ kernel_execution_control_mode = atof(std::getenv("SELECTIVEX_KERNEL_EXECUTION_CONTROL_MODE")); }
  cp_costs_ref[0]=0;
  for (auto& it : model_container){
    comp_kernel_ref_map[it.first] = active_kernels[it.second.val_index];
    if (comp_kernel_save_map.find(it.first) != comp_kernel_save_map.end()){
      active_kernels[it.second.val_index] = comp_kernel_save_map[it.first];
    } else{
      reset_kernel(active_kernels[it.second.val_index]);
      if (reset_kernel_execution_state){
        active_kernels[it.second.val_index].is_steady = false;
        active_kernels[it.second.val_index].is_active = false;
      }
      clear_kernel_distribution(active_kernels[it.second.val_index]);
    }
  }
}
*/

void finalize(){
  // 'spf' deletion should occur automatically at program shutdown
  for (auto it : aggregate_channel_map){
    free(it.second);
  }
  if (std::getenv("SELECTIVEX_VIZ_FILE") != NULL){
    if (is_world_root){
      stream.close();
      stream_tune.close();
      stream_reconstruct.close();
      stream_kernel.close();
    }
  }
}

}
}
}
