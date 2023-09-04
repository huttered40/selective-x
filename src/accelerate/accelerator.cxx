#include "accelerator.h"
#include "container/symbol_tracker.h"
#include "util/util.h"
#include "../util/util.h"

namespace selectivex{
namespace internal{
namespace accelerate{

static void update_path_profile(float* in, float* inout, size_t len){
  assert(len == cp_costs_size);	// this assert prevents user from obtaining wrong output if MPI implementation cuts up the message.
  // Depending on which process has larger estimated {kernel,total} execution time, update the rest of the path profile (floats at indices aux_info_size+)
  // The rest of the path profile (floats at indices aux_info_size+) will store kernel information executed along the current (sub-)critical execution path.
  if (in[0] > inout[0]){
    std::memcpy(inout+aux_info_size,in+aux_info_size,(len-aux_info_size)*sizeof(float));
  }
  for (auto i=0; i<aux_info_size; i++) inout[i] = std::max(in[i],inout[i]);
}
static void propagate_cp_op(float* in, float* inout, int* len, MPI_Datatype* dtype){
  update_path_profile(in,inout,static_cast<size_t>(*len));
}

/* TODO: Excavate later.
static void kernel_update(float* read_ptr){
  // Leave the un-updated kernels alone, just use the per-process count
  // Lets have all processes update, even the root, so that they leave this routine (and subsequently leave the interception) at approximately the same time.
  for (auto i=0; i<comp_kernel_select_count; i++){
    //TODO: Why the 9? I guess it is 9 floats, all necessary to uniquely identify a computational kernel.
    auto offset = i*9;
    if (read_ptr[offset] == -1) break;
    comp_kernel_key id(-1,(int)read_ptr[offset],read_ptr[offset+7],
                          (int)read_ptr[offset+1],(int)read_ptr[offset+2],(int)read_ptr[offset+3],
                          (int)read_ptr[offset+4],(int)read_ptr[offset+5]);
    // Don't bother adding new kernels unseen by the current processor.
    // Kernels un-updated will use per-process count as an approximation
    if (comp_kernel_map.find(id) != comp_kernel_map.end()){
      //TODO: 'num_local_schedules' along critical-path? or along a processor's execution path?
      active_kernels[comp_kernel_map[id].val_index].num_local_schedules = read_ptr[offset+8];
    }
  }
  for (auto i=0; i<comm_kernel_select_count; i++){
    auto offset = comp_kernel_select_count*9+i*9;
    if (read_ptr[offset] == -1) break;
    comm_kernel_key id(-1,(int)read_ptr[offset],(int*)&read_ptr[offset+1],
                       (int*)&read_ptr[offset+3],read_ptr[offset+7],
                       (int)read_ptr[offset+5]); 
    // Don't bother adding new kernels unseen by the current processor.
    // Kernels un-updated will use per-process count as an approximation
    if (comm_kernel_map.find(id) != comm_kernel_map.end()){
      active_kernels[comm_kernel_map[id].val_index].num_local_schedules = read_ptr[offset+8];
    }
  }
}
*/

void accelerator::exchange_communicators(MPI_Comm oldcomm, MPI_Comm newcomm){
  // Save and accumulate the computation time between last communication routine as both execution-time and computation time
  //   into both the execution-time critical path data structures and the per-process data structures.
  auto save_comp_time = MPI_Wtime() - computation_timer;
  if (mode==1 && measure_kernel_time==0) cp_costs[0] += save_comp_time;

  //TODO: Cool concept, but we likely don't need to do this anymore. Just too costly.
  //generate_aggregate_channels(oldcomm,newcomm);
  PMPI_Barrier(oldcomm);
  if (mode==1) computation_timer = MPI_Wtime();
}

template <size_t nfeatures>
bool accelerate::should_observe(std:string kernel_name, volatile double curtime, float* features, MPI_Comm cm){
  // Save and accumulate the computation time between last communication routine as both execution-time and computation time
  //   into both the execution-time critical path data structures and the per-process data structures.
  volatile auto overhead_start_time = MPI_Wtime();
  // At this point, 'cp_costs[0]' tells me the process's time up until now. A barrier won't suffice when kernels are conditionally scheduled.
  if (measure_kernel_time==0) cp_costs[0] += (curtime - computation_timer;);
  // Special exit if no kernels are to be scheduled -- the goal is to track the total overhead time (no comp/comm kernels), which should
  //   be attained with timers outside of selectivex.
  if (schedule_kernels==0){ return false; }// Don't need to update 'intercept_overhead' in this case (assume 0)

  // If the kernel has not been "intercepted" or "invoked" before, then save it.
  if (kernel_id_map.find(kernel_name) == kernel_id_map.end()) kernel_id_map[kernel_name]=kernel_id_count++;
  auto kernel_id = kernel_id_map[kernel_name];
  if (model_containers.find(kernel_id) == model_containers.end()) model_containers[kernel_id]=kernel<nfeatures>(kernel_name,kernel_id);


  if (cm==MPI_COMM_NULL) schedule_decision = model_containers[kernel_id].is_active(features);
  else{
    memset(&cp_costs[1],0,(aux_info_size-1)*sizeof(float)); cp_costs[3] = 1;
    if (comm_kernel_map.find(key) != comm_kernel_map.end()){
      // should_observe(...) is equivalent to asking whether a kernel is globally steady.
      // Globally-steady (communication) kernels don't require synchronization to ask whether to invoke.
      // However, communication is still necessary to propagate approximate execution time (which limits speedups).
      schedule_decision = should_observe(comm_kernel_map[key])==1;
      cp_costs[2] = (!schedule_decision ? 1 : 0);// set to 1 if kernel is globally steady.
      cp_costs[3] = (schedule_decision ? 1 : 0);// set to 0 if kernel is globally steady.
      if (!schedule_decision){
	// If this particular kernel is globally steady, meaning it has exhausted its state aggregation channels,
	//   then we can overwrite the '-1' with the sample mean of the globally-steady kernel
	cp_costs[1] = active_kernels[comm_kernel_map[key].val_index].is_steady;
      }
    }
    // Use pathsets, not batches, to check if kernel can leverage an aggregation. Such a kernel must be locally steady (i.e.
    //   from its own schedules, its steady), and must be able to aggregate across the channel associated with 'tracker.comm'
    if ((tracker.partner1 == -1) && (propagate_kernel_execution_state>0)){
      for (auto& it : comp_kernel_map){
	if (!((active_kernels[it.second.val_index].is_steady==1) && (should_observe(it.second)==1))) continue;
	// Any global communicator can fast-track a communication kernel to being in global steady state. No need to match up hashes (that would only be necessary for sample aggregation)
	if (aggregate_channel_map[comm_channel_map[tracker.comm]->global_hash_tag]->is_final){
	  tracker.save_comp_key.push_back(it.first);
	  cp_costs[4]++;
	}
	else{
	  if (active_kernels[it.second.val_index].registered_channels.find(comm_channel_map[tracker.comm]) != active_kernels[it.second.val_index].registered_channels.end()) continue;
	  // TODO: Not exactly sure whether to use global_hash_id below or local_hash_id
	  if (aggregate_channel_map.find(active_kernels[it.second.val_index].hash_id ^ aggregate_channel_map[comm_channel_map[tracker.comm]->global_hash_tag]->global_hash_tag) == aggregate_channel_map.end()) continue;
	  tracker.save_comp_key.push_back(it.first);
	  cp_costs[4]++;
	}
      }
    }
    if ((tracker.partner1 == -1) && (propagate_kernel_execution_state>0)){
      for (auto& it : comm_kernel_map){
	if (!((active_kernels[it.second.val_index].is_steady==1) && (should_observe(it.second)==1))) continue;
	if (aggregate_channel_map[comm_channel_map[tracker.comm]->global_hash_tag]->is_final){
	  tracker.save_comm_key.push_back(it.first);
	  cp_costs[5]++;
	}
	else{
	  if (active_kernels[it.second.val_index].registered_channels.find(comm_channel_map[tracker.comm]) != active_kernels[it.second.val_index].registered_channels.end()) continue;
	  // TODO: Not exactly sure whether to use global_hash_id below or local_hash_id
	  if (aggregate_channel_map.find(active_kernels[it.second.val_index].hash_id ^ aggregate_channel_map[comm_channel_map[tracker.comm]->global_hash_tag]->global_hash_tag) == aggregate_channel_map.end()) continue;
	  tracker.save_comm_key.push_back(it.first);
	  cp_costs[5]++;
	}
      }
    }
    if (partner1 == -1){
      MPI_Op op_special = MPI_MAX;
      if (kernel_execution_count_mode == 2) MPI_Op_create((MPI_User_function*) propagate_cp_op,0,&op_special);
      PMPI_Allreduce(MPI_IN_PLACE, &cp_costs[0], cp_costs.size(), MPI_FLOAT, op_special, tracker.comm);
      if (kernel_execution_count_mode == 2) MPI_Op_free(&op_special);
      if (collective_state_protocol) schedule_decision = (cp_costs[3] == 0 ? false : true);
      else schedule_decision = (cp_costs[2] == 0 ? true : false);
      tracker.aggregate_comp_kernels = cp_costs[4]>0;
      tracker.aggregate_comm_kernels = cp_costs[5]>0;
      tracker.should_propagate = cp_costs[4]>0 || cp_costs[5]>0;
      if (comm_kernel_map.find(key) != comm_kernel_map.end()){
	if (!schedule_decision){
	  set_kernel_state(comm_kernel_map[key],false);
	  set_kernel_state_global(comm_kernel_map[key],false);
	}
      }
    }
    else{
      bool has_received=false;
      if ((is_sender) && (rank != partner1)){
	MPI_Buffer_attach(&eager_pad[0],eager_pad.size());
	PMPI_Bsend(&cp_costs[0], cp_costs.size(), MPI_FLOAT, partner1, internal_tag2, tracker.comm);
	void* temp_buf; int temp_size;
	MPI_Buffer_detach(&temp_buf,&temp_size);
      }
      if ((!is_sender) && (rank != partner1)){
	has_received=true;
	PMPI_Recv(&cp_costs_foreign[0], cp_costs_foreign.size(), MPI_FLOAT, partner1, internal_tag2, tracker.comm, MPI_STATUS_IGNORE);
      }
      if ((partner2 != -1) && (rank != partner2)){
	has_received=true;
	PMPI_Recv(&cp_costs_foreign[0], cp_costs_foreign.size(), MPI_FLOAT, partner2, internal_tag2, tracker.comm, MPI_STATUS_IGNORE);
      }
      if (has_received){
	schedule_decision = (cp_costs_foreign[2] == 0);
	update_path_profile(&cp_costs_foreign[0],&cp_costs[0],cp_costs_size);
	if (tracker.tag >= 13 && tracker.tag <= 14) schedule_decision = (cp_costs[2] == 0);
	if (comm_kernel_map.find(key) != comm_kernel_map.end()){
	  if (!schedule_decision){
	    set_kernel_state(comm_kernel_map[key],false);
	    set_kernel_state_global(comm_kernel_map[key],false);
	  }
	}
      }
    }
    // Only Receivers always update their kernel maps
    if (kernel_execution_count_mode == 2){
      if (!(tracker.partner1 != -1 && tracker.is_sender && tracker.partner2 == tracker.partner1)) kernel_update(&cp_costs_foreign[aux_info_size]);
    }
  }

  // Do not allow interception of communication routines until the ensuing kernel is finished executing.
  // A primary use case here is personalized collectives. It is much more efficient to let NOT intercept these and incorporate into
  //   a higher-level kernel. The only downside is that explicit information is needed from the user to enable this: they must call
  //   bool decision = should_observe("kernel",..); double t=MPI_Wtime(); if (decision) { ... } t=MPI_Wtime()-t; observe("kernel",..);
  // We will allow dedicated BLAS/LAPACK kernels to be intercepted within user-defined kernels,
  //   but note that there is the caution of issues with timer explicit/implicit
  //   and/or added overhead of checking whether those nested computation routines can be avoided.
  // This is only a concern for dedicated kernels (MPI,BLAS,LAPACK), as no user would do this manually.
  within_kernel=1;// signifies that execution is about to invoke a kernel
  // start timer for computation routine
  intercept_overhead += MPI_Wtime() - overhead_start_time;
  return schedule_decision;
}

template <size_t nfeatures>
void accelerate::observe(std::string kernel_name, volatile double kernel_time, float* features, MPI_Comm cm){
  // Special exit if no kernels are to be scheduled -- the goal is to track the total overhead time (no comp/comm kernels), which should
  //   be attained with timers outside of selectivex.
  if (schedule_kernels==0){ return; }
  volatile auto overhead_start_time = MPI_Wtime();

  assert(kernel_id_map.find(kernel_name) != kernel_id_map.end());
  auto kernel_per_input<nfeatures> dummy(features);// just instantiate one, then check if it exists already
  // For debugging sanity, if the autotuner shut off execution of this kernel, then have debugger use only existing mean
  //TODO: What if kernel_time is bad (kernel wasn't scheduled)?
  // Note: a kernel's stats and its state are not equivalent. Below does not update its state.
  model_containers[kernel_id].update_model(features,kernel_time);
  // Note: 'get_estimate' must be called before setting the updated kernel state. If kernel was not scheduled, kernel_time set below overwrites 'kernel_time'
  //TODO: Consider always using the model prediction rather than the true time, even if the kernel was active?
  kernel_time = model_containers[kernel_id].is_active(features) == false ? model_containers[kernel_id].get_estimate(features) : kernel_time;
  if (cm==MPI_COMM_NULL) model_containers[kernel_id].set_active(features);// set to whether underlying distribution is "steady"
  else{
    if (tracker.partner1 == -1){
      bool _is_steady = is_steady(key,comm_kernel_map[key]);
      set_kernel_state(comm_kernel_map[key],!_is_steady);
      if (kernel_execution_count_mode == -1) set_kernel_state_global(comm_kernel_map[key],!_is_steady);// Force global state to steady.
      if (propagate_kernel_execution_state == 0) { set_kernel_state_global(comm_kernel_map[key],!_is_steady); }
    } else{
      // If p2p (most notably senders) must delay update, we must check whether the kernel has already been set to local steady state
      if (delay_state_update){
	if (active_kernels[comm_kernel_map[key].val_index].is_steady){
	  set_kernel_state_global(comm_kernel_map[key],false);
	} else{
	  bool _is_steady = is_steady(key,comm_kernel_map[key]);
	  set_kernel_state(comm_kernel_map[key],!_is_steady);
	}
      } else{
	bool _is_steady = is_steady(key,comm_kernel_map[key]);
	set_kernel_state(comm_kernel_map[key],!_is_steady);
	if (kernel_execution_count_mode == -1) set_kernel_state_global(comm_kernel_map[key],!_is_steady);// Force global state to steady.
	set_kernel_state_global(comm_kernel_map[key],!_is_steady);
      }
    }
  }

  schedule_decision = false;
  within_kernel=0;// signifies that execution has left a kernel
  cp_costs[0] += kernel_time;
  intercept_overhead += MPI_Wtime() - overhead_start_time;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

bool accelerator::initiate_comm(blocking& tracker, volatile double curtime, int64_t nelem, MPI_Datatype t, MPI_Comm comm,
                         bool is_sender, int partner1, int user_tag1, int partner2, int user_tag2){
  //TODO: Excavate the below later. It is used to generate the right features. It should be built into intercept/comm.cxx
/*
  // Save caller communication attributes into reference object for use in corresponding static method 'complete_comm'
  int rank; MPI_Comm_rank(comm, &rank);
  int word_size,np; MPI_Type_size(t, &word_size);
  int64_t nbytes = word_size * nelem;
  MPI_Comm_size(comm, &np);
  tracker.nbytes = nbytes;
  tracker.comm = comm;
  tracker.comm_size = np;
  tracker.is_sender = is_sender;
  tracker.partner1 = partner1;
  tracker.partner2 = partner2 != -1 ? partner2 : partner1;// Useful in propagation
  //TODO: What do these three help with?
  tracker.should_propagate = false;
  tracker.aggregate_comp_kernels=false;
  tracker.aggregate_comm_kernels=false;
  for (auto i=0; i<comm_channel_map[tracker.comm]->id.size(); i++){
    comm_sizes[i]=comm_channel_map[tracker.comm]->id[i].first;
    comm_strides[i]=comm_channel_map[tracker.comm]->id[i].second;
  }
  // Below, the idea is that key doesn't exist in comm_kernel_map iff the key hasn't been seen before. If the key has been seen, we automatically
  //   create an entry in comm_kernel_key, although it will be empty.
  comm_kernel_key key(rank,-1,tracker.tag,comm_sizes,comm_strides,tracker.nbytes,tracker.partner1);
*/
/*
  //TODO: Is there a way to avoid this explicit copy? We want to leverage the order of the arrays, right?
  //TODO: This would never be practical if the number of distinct kernels is > 10 or 20.
  if (kernel_execution_count_mode == 2){
    // Fill in -1 first because the number of distinct kernels might be less than 'comm_kernel_select_count',
    //   just to avoid confusion. A -1 tag clearly means that the kernel is void
    memset(&cp_costs[aux_info_size],-1,sizeof(float)*(cp_costs.size()-aux_info_size));
    // Iterate over first 'comp_kernel_select_count' keys
    int count=0;
    for (auto it : comp_kernel_map){
      if (comp_kernel_select_count==0) break;
      if (active_kernels[it.second.val_index].num_local_schedules == 0) break;
      auto offset = aux_info_size+9*count;
      cp_costs[offset] = it.first.tag;
      cp_costs[offset+1] = it.first.param1;
      cp_costs[offset+2] = it.first.param2;
      cp_costs[offset+3] = it.first.param3;
      cp_costs[offset+4] = it.first.param4;
      cp_costs[offset+5] = it.first.param5;
      cp_costs[offset+6] = it.first.kernel_index;
      cp_costs[offset+7] = it.first.flops;
      cp_costs[offset+8] = active_kernels[it.second.val_index].num_local_schedules;
      count++; if (count==comp_kernel_select_count) break;
    }
    count=0;
    for (auto it : comm_kernel_map){
      if (comm_kernel_select_count==0) break;
      if (active_kernels[it.second.val_index].num_local_schedules == 0) break;
      auto offset = aux_info_size+9*comp_kernel_select_count+count*9;
      cp_costs[offset] = it.first.tag;
      cp_costs[offset+1] = it.first.dim_sizes[0];
      cp_costs[offset+2] = it.first.dim_sizes[1];
      cp_costs[offset+3] = it.first.dim_strides[0];
      cp_costs[offset+4] = it.first.dim_strides[1];
      cp_costs[offset+5] = it.first.partner_offset;
      cp_costs[offset+6] = it.first.kernel_index;
      cp_costs[offset+7] = it.first.msg_size;
      cp_costs[offset+8] = active_kernels[it.second.val_index].num_local_schedules;
      count++; if (count==comm_kernel_select_count) break;
    }
  }
*/

}

// Used only for p2p communication. All blocking collectives use sychronous protocol
void accelerator::complete_comm(blocking& tracker){
// Old notes
/*
      // Note if this is true, the corresponding entry in the batch map must be cleared. However, I think I delete the entire map in aggregation mode 1, so asserting
      //   on this is difficult.
      // Note: I think branching on aggregation mode is not needed. The pathset should contain all batch samples and the batch should be cleared.

    // Propogate critical paths for all processes in communicator based on what each process has seen up until now (not including this communication)
    if (tracker.should_propagate && tracker.partner1 == -1){
      bool is_world_communication = (tracker.comm == MPI_COMM_WORLD) && (tracker.partner1 == -1);
      if ((rank == tracker.partner1) && (rank == tracker.partner2)) { ; }
      else{
	//if (tracker.aggregate_comm_kernels) comm_state_aggregation(tracker);
	//if (tracker.aggregate_comp_kernels) comp_state_aggregation(tracker);
      }
    }
    tracker.should_propagate = false;
    tracker.aggregate_comp_kernels = false;
    tracker.aggregate_comm_kernels = false;

  computation_timer = MPI_Wtime();
*/
}

/*
// Called by both nonblocking p2p and nonblocking collectives
bool accelerator::initiate_comm(nonblocking& tracker, volatile double curtime, int64_t nelem,
                         MPI_Datatype t, MPI_Comm comm, bool is_sender, int partner, int user_tag){
  // At this point, 'cp_costs' tells me the process's time up until now. A barrier won't suffice when kernels are conditionally scheduled.
  int rank; MPI_Comm_rank(comm, &rank);
  volatile auto overhead_start_time = MPI_Wtime();
  // Save caller communication attributes into reference object for use in corresponding static method 'complete_comm'
  int word_size,np; MPI_Type_size(t, &word_size);
  int64_t nbytes = word_size * nelem;
  MPI_Comm_size(comm, &np);
  tracker.nbytes = nbytes;
  tracker.comm = comm;
  tracker.comm_size = np;
  tracker.is_sender = is_sender;
  tracker.partner1 = partner;
  tracker.partner2 = partner;
  tracker.should_propagate = false;
  tracker.aggregate_comp_kernels=false;
  tracker.aggregate_comm_kernels=false;

  schedule_decision = true;
  // Assume that the communicator of either collective/p2p is registered via comm_split, and that its described using a max of 3 dimension tuples.
  assert(comm_channel_map.find(tracker.comm) != comm_channel_map.end());
  int comm_sizes[2]={0,0}; int comm_strides[2]={0,0};
  for (auto i=0; i<comm_channel_map[tracker.comm]->id.size(); i++){
    comm_sizes[i]=comm_channel_map[tracker.comm]->id[i].first;
    comm_strides[i]=comm_channel_map[tracker.comm]->id[i].second;
  }
  // Below, the idea is that key doesn't exist in comm_kernel_map iff the key hasn't been seen before. If the key has been seen, we automatically
  //   create an entry in comm_kernel_key, although it will be empty.
  comm_kernel_key key(rank,-1,tracker.tag,comm_sizes,comm_strides,tracker.nbytes,tracker.partner1);
  memset(&cp_costs[1],0,(aux_info_size-1)*sizeof(float)); cp_costs[3] = 1;
  if (comm_kernel_map.find(key) != comm_kernel_map.end()){
    schedule_decision = should_observe(comm_kernel_map[key])==1;
    cp_costs[2] = (!schedule_decision ? 1 : 0);	// This logic must match that in 'initiate_comm(blocking&,...)'
    cp_costs[3] = (schedule_decision ? 1 : 0);	// This logic must match that in 'initiate_comm(blocking&,...)'
    if (!schedule_decision){
      // If this particular kernel is globally steady, meaning it has exhausted its state aggregation channels,
      //   then we can overwrite the '-1' with the sample mean of the globally-steady kernel
      cp_costs[1] = active_kernels[comm_kernel_map[key].val_index].is_steady;
    }
  }

  // Note: I will not write special case for rank==partner
  if (kernel_execution_count_mode == 2){
    // Fill in -1 first because the number of distinct kernels might be less than 'comm_kernel_select_count',
    //   just to avoid confusion. A -1 tag clearly means that the kernel is void
    memset(&cp_costs[aux_info_size],-1,sizeof(float)*(cp_costs.size()-aux_info_size));
    // Iterate over first 'comp_kernel_select_count' keys
    int count=0;
    for (auto it : comp_kernel_map){
      if (comp_kernel_select_count==0) break;
      if (active_kernels[it.second.val_index].num_local_schedules == 0) break;
      auto offset = aux_info_size+9*count;
      cp_costs[offset] = it.first.tag;
      cp_costs[offset+1] = it.first.param1;
      cp_costs[offset+2] = it.first.param2;
      cp_costs[offset+3] = it.first.param3;
      cp_costs[offset+4] = it.first.param4;
      cp_costs[offset+5] = it.first.param5;
      cp_costs[offset+aux_info_size] = it.first.kernel_index;
      cp_costs[offset+7] = it.first.flops;
      cp_costs[offset+8] = active_kernels[it.second.val_index].num_local_schedules;
      count++; if (count==comp_kernel_select_count) break;
    }
    count=0;
    for (auto it : comm_kernel_map){
      if (comm_kernel_select_count==0) break;
      if (active_kernels[it.second.val_index].num_local_schedules == 0) break;
      auto offset = aux_info_size+9*comp_kernel_select_count+count*9;
      cp_costs[offset] = it.first.tag;
      cp_costs[offset+1] = it.first.dim_sizes[0];
      cp_costs[offset+2] = it.first.dim_sizes[1];
      cp_costs[offset+3] = it.first.dim_strides[0];
      cp_costs[offset+4] = it.first.dim_strides[1];
      cp_costs[offset+5] = it.first.partner_offset;
      cp_costs[offset+6] = it.first.kernel_index;
      cp_costs[offset+7] = it.first.msg_size;
      cp_costs[offset+8] = active_kernels[it.second.val_index].num_local_schedules;
      count++; if (count==comm_kernel_select_count) break;
    }
  }
  save_path_data = nullptr; save_prop_req = MPI_REQUEST_NULL;
  if (tracker.partner1 == -1){
    assert(delay_state_update);
    save_path_data = (float*)malloc(cp_costs_size*sizeof(float));
    std::memcpy(save_path_data,&cp_costs[0],cp_costs_size*sizeof(float));
    MPI_Op op = MPI_MAX;
    if (kernel_execution_count_mode == 2)MPI_Op_create((MPI_User_function*) propagate_cp_op,0,&op);
    PMPI_Iallreduce(MPI_IN_PLACE, save_path_data, cp_costs_size, MPI_FLOAT, op, tracker.comm, &save_prop_req);
    //MPI_Op_free(&op);
  }
  else{
    if (is_sender){
      MPI_Buffer_attach(&eager_pad[0],eager_pad.size());
      PMPI_Bsend(&cp_costs[0],cp_costs_size,MPI_FLOAT,tracker.partner1,internal_tag2,tracker.comm);
      void* temp_buf; int temp_size;
      MPI_Buffer_detach(&temp_buf,&temp_size);
    } else{
      assert(delay_state_update);
      save_path_data = (float*)malloc(cp_costs_size*sizeof(float));
      PMPI_Irecv(save_path_data, cp_costs_size, MPI_FLOAT, tracker.partner1, internal_tag2, tracker.comm, &save_prop_req);
    }
  }
  if (!schedule_decision){
    while (1){
      if (request_id == INT_MAX) request_id = 100;// reset to avoid overflow. rare case.
      if ((nonblocking_internal_info.find(request_id) == nonblocking_internal_info.end()) && (request_id != MPI_REQUEST_NULL)){
        nonblocking_info msg_info(save_path_data,save_prop_req,false,is_sender,partner,comm,(float)nbytes,(float)np,user_tag,&tracker);
        nonblocking_internal_info[request_id] = msg_info;
        break;
      }
      request_id++;
    }
  }

  intercept_overhead += MPI_Wtime() - overhead_start_time;
  return schedule_decision;
}

// Called by both nonblocking p2p and nonblocking collectives
void accelerator::initiate_comm(nonblocking& tracker, volatile double itime, int64_t nelem,
                         MPI_Datatype t, MPI_Comm comm, MPI_Request* request, bool is_sender, int partner, int user_tag){
  // Deal with computational cost at the beginning, but don't synchronize to find computation-critical path-path yet or that will screw up calculation of overlap!
  if (measure_kernel_time==0) cp_costs[0] += itime;

  int el_size,p;
  MPI_Type_size(t, &el_size);
  int64_t nbytes = el_size * nelem;
  MPI_Comm_size(comm, &p);

  // Note: I will not write special case for rank==partner
  // These asserts are to prevent the situation in which my synthetic request_id and that of the MPI implementation collide
  assert(nonblocking_internal_info.find(*request) == nonblocking_internal_info.end());
  nonblocking_info msg_info(save_path_data,save_prop_req,true,is_sender,partner,comm,(float)nbytes,(float)p,user_tag,&tracker);
  nonblocking_internal_info[*request] = msg_info;
}

void accelerator::complete_comm(nonblocking& tracker, float* path_data, MPI_Request* request, double comp_time, double comm_time){
  auto info_it = nonblocking_internal_info.find(*request);
  assert(info_it != nonblocking_internal_info.end());

  tracker.is_sender = info_it->second.is_sender;
  tracker.comm = info_it->second.comm;
  tracker.partner1 = info_it->second.partner;
  tracker.partner2 = -1;
  tracker.nbytes = info_it->second.nbytes;
  tracker.comm_size = info_it->second.comm_size;

  int rank; MPI_Comm_rank(tracker.comm,&rank);
  // Right now, I need to check schedule_decision and replace 'comm_time' with the predicted time, if necessary.
  int comm_sizes[2]={0,0}; int comm_strides[2]={0,0};
  for (auto i=0; i<comm_channel_map[tracker.comm]->id.size(); i++){
    comm_sizes[i]=comm_channel_map[tracker.comm]->id[i].first;
    comm_strides[i]=comm_channel_map[tracker.comm]->id[i].second;
  }
  // Below, the idea is that key doesn't exist in comm_kernel_map iff the key hasn't been seen before. If the key has been seen, we automatically
  //   create an entry in comm_kernel_key, although it will be empty.
  comm_kernel_key key(rank,active_kernels.size(),tracker.tag,comm_sizes,comm_strides,tracker.nbytes,tracker.partner1);
  if (comm_kernel_map.find(key) == comm_kernel_map.end()){
    active_comm_kernel_keys.push_back(key);
    if (tracker.partner1 != -1){
      auto world_partner_rank = channel::translate_rank(tracker.comm,tracker.partner1);
      active_kernels.emplace_back();
    } else{
      assert(comm_channel_map.find(tracker.comm) != comm_channel_map.end());
      active_kernels.emplace_back(comm_channel_map[tracker.comm]);
    }
    comm_kernel_map[key] = kernel_key_id(true,active_comm_kernel_keys.size()-1,active_kernels.size()-1,false);
  }
  // For debugging sanity, if the autotuner shut off execution of this kernel, then have debugger use only existing mean
  update_model(comm_kernel_map[key],comm_time);
  if (should_observe(comm_kernel_map[key])==0){
    // Note if this is true, the corresponding entry in the batch map must be cleared. However, I think I delete the entire map in aggregation mode 1, so asserting
    //   on this is difficult.
    // Note: I think branching on aggregation mode is not needed. The pathset should contain all batch samples and the batch should be cleared.
    comm_time = get_estimate(comm_kernel_map[key]);
  } else{
    if (delay_state_update){
      // Receivers of any kind must transfer the sender's local state first. This then allows a quick jump from active to globally steady
      if (tracker.partner1 == -1 || !tracker.is_sender) set_kernel_state(comm_kernel_map[key],path_data[4]==0);
      if (active_kernels[comm_kernel_map[key].val_index].is_steady){
	set_kernel_state_global(comm_kernel_map[key],false);
      } else{
	bool is_steady = is_steady(key,comm_kernel_map[key]);
	set_kernel_state(comm_kernel_map[key],!is_steady);
      }
    } else{
      bool is_steady = is_steady(key,comm_kernel_map[key]);
      set_kernel_state(comm_kernel_map[key],!is_steady);
      if (kernel_execution_count_mode == -1) set_kernel_state_global(comm_kernel_map[key],!is_steady);// Force global state to steady.
      set_kernel_state_global(comm_kernel_map[key],!is_steady);
    }
  }

  cp_costs[0] += comm_time;		// update critical path runtime
  if (measure_kernel_time==0) cp_costs[0] += comp_time;

  if (!tracker.is_sender || tracker.partner1==-1) free(info_it->second.path_data);
  nonblocking_internal_info.erase(*request);
}

int accelerator::complete_comm(double curtime, MPI_Request* request, MPI_Status* status){
  auto comp_time = curtime - computation_timer;
  int ret = MPI_SUCCESS;
  assert(nonblocking_internal_info.find(*request) != nonblocking_internal_info.end());
  auto info_it = nonblocking_internal_info.find(*request);
  auto save_r = info_it->first;
  int rank; MPI_Comm_rank(info_it->second.track->comm,&rank);
  if (info_it->second.is_active == 1){
    volatile auto last_start_time = MPI_Wtime();
    ret = PMPI_Wait(request, status);
    auto save_comm_time = MPI_Wtime() - last_start_time;
    auto overhead_start_time = MPI_Wtime();
    // If receiver or collective, complete the barrier and the path data propagation
    if (!info_it->second.is_sender || info_it->second.partner==-1){
      assert(info_it->second.path_data != nullptr);
      PMPI_Wait(&info_it->second.prop_req, MPI_STATUS_IGNORE);
      if (info_it->second.partner != -1) update_path_profile(info_it->second.path_data,&cp_costs[0],cp_costs_size);
      //if (kernel_execution_count_mode == 2) kernel_update(&info_it->second.path_data[aux_info_size]);
    }
    complete_comm(*info_it->second.track, info_it->second.path_data, &save_r, comp_time, save_comm_time);
    intercept_overhead += MPI_Wtime() - overhead_start_time;
  } else{
    auto overhead_start_time = MPI_Wtime();
    // If receiver or collective, complete the barrier and the path data propagation
    if (!info_it->second.is_sender || info_it->second.partner==-1){
      assert(info_it->second.path_data != nullptr);
      PMPI_Wait(&info_it->second.prop_req, MPI_STATUS_IGNORE);
      if (info_it->second.partner != -1) update_path_profile(info_it->second.path_data,&cp_costs[0],cp_costs_size);
      //if (kernel_execution_count_mode == 2) kernel_update(&info_it->second.path_data[aux_info_size]);
      if (status != MPI_STATUS_IGNORE){
        status->MPI_SOURCE = info_it->second.partner;
        status->MPI_TAG = info_it->second.tag;
      }
    }
    complete_comm(*info_it->second.track, info_it->second.path_data, &save_r, comp_time, 1000000.);
    *request = MPI_REQUEST_NULL;
    intercept_overhead += MPI_Wtime() - overhead_start_time;
  }
  return ret;
}

int accelerator::complete_comm(double curtime, int count, MPI_Request array_of_requests[], int* indx, MPI_Status* status){
  auto comp_time = curtime - computation_timer;
  auto overhead_start_time = MPI_Wtime();
  int ret = MPI_SUCCESS;
  std::vector<MPI_Request> pt(count);
  int num_skips=0; int last_skip;
  for (int i=0;i<count;i++){
    assert(nonblocking_internal_info.find((array_of_requests)[i]) == nonblocking_internal_info.end());
    if (nonblocking_internal_info[(array_of_requests)[i]].is_active){
      pt[i] = (array_of_requests)[i];
    } else{
      pt[i] = MPI_REQUEST_NULL; num_skips++; last_skip = i;
    }
  }
  intercept_overhead += MPI_Wtime() - overhead_start_time;
  if (num_skips < pt.size()){
    volatile auto last_start_time = MPI_Wtime();
    ret = PMPI_Waitany(count,array_of_requests,indx,status);
    auto save_comm_time = MPI_Wtime() - last_start_time;
    overhead_start_time = MPI_Wtime();
    auto info_it = nonblocking_internal_info.find((array_of_requests)[*indx]);
    assert(info_it != nonblocking_internal_info.end());
    int rank; MPI_Comm_rank(info_it->second.track->comm,&rank);
    // If receiver or collective, complete the barrier and the path data propagation
    if (!info_it->second.is_sender || info_it->second.partner==-1){
      assert(info_it->second.path_data != nullptr);
      PMPI_Wait(&info_it->second.prop_req, MPI_STATUS_IGNORE);
      if (info_it->second.partner != -1) update_path_profile(info_it->second.path_data,&cp_costs[0],cp_costs_size);
      //if (kernel_execution_count_mode == 2) kernel_update(&info_it->second.path_data[aux_info_size]);
    }
    complete_comm(*info_it->second.track, info_it->second.path_data, &(array_of_requests)[*indx], comp_time, save_comm_time);
  } else{
    overhead_start_time = MPI_Wtime();
    auto info_it = nonblocking_internal_info.find((array_of_requests)[last_skip]);
    assert(info_it != nonblocking_internal_info.end());
    int rank; MPI_Comm_rank(info_it->second.track->comm,&rank);
    // If receiver or collective, complete the barrier and the path data propagation
    if (!info_it->second.is_sender || info_it->second.partner==-1){
      assert(info_it->second.path_data != nullptr);
      PMPI_Wait(&info_it->second.prop_req, MPI_STATUS_IGNORE);
      if (info_it->second.partner != -1) update_path_profile(info_it->second.path_data,&cp_costs[0],cp_costs_size);
      //if (kernel_execution_count_mode == 2) kernel_update(&info_it->second.path_data[aux_info_size]);
      if (status != MPI_STATUS_IGNORE){
        status->MPI_SOURCE = info_it->second.partner;
        status->MPI_TAG = info_it->second.tag;
      }
    }
    complete_comm(*info_it->second.track, info_it->second.path_data, &(array_of_requests)[last_skip], comp_time, 1000000.);
    (array_of_requests)[last_skip] = MPI_REQUEST_NULL;
  }
  intercept_overhead += MPI_Wtime() - overhead_start_time;
  return ret;
}

int accelerator::complete_comm(double curtime, int incount, MPI_Request array_of_requests[], int* outcount, int array_of_indices[],
                        MPI_Status array_of_statuses[]){
  int indx; MPI_Status stat;
  int ret = complete_comm(curtime,incount,array_of_requests,&indx,&stat);
  if (array_of_statuses != MPI_STATUSES_IGNORE) array_of_statuses[indx] = stat;
  array_of_indices[0] = indx;
  *outcount=1;
  return ret;
}

int accelerator::complete_comm(double curtime, int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]){
  auto comp_time = curtime - computation_timer;
  auto overhead_start_time = MPI_Wtime();
  int ret = MPI_SUCCESS;
  int true_count = count;
  std::vector<MPI_Request> pt(count); for (int i=0;i<count;i++){pt[i]=(array_of_requests)[i];}
  // Scan over the requests to identify those that are 'fake'
  for (int i=0; i<count; i++){
    MPI_Request request = array_of_requests[i];
    // 1. check if this request is fake. If so, update its status if a receiver and set its request to MPI_REQUEST_NULL and decrement a count
    auto info_it = nonblocking_internal_info.find(request);
    assert(info_it != nonblocking_internal_info.end());
    schedule_decision = info_it->second.is_active;
    if (schedule_decision == 0){
      true_count--;
      int rank; MPI_Comm_rank(info_it->second.track->comm,&rank);
      if (rank != info_it->second.partner){
        // If receiver or collective, complete the barrier and the path data propagation
        if (!info_it->second.is_sender || info_it->second.partner==-1){
          assert(info_it->second.path_data != nullptr);
          PMPI_Wait(&info_it->second.prop_req, MPI_STATUS_IGNORE);
          update_path_profile(info_it->second.path_data,&cp_costs[0],cp_costs_size);
          //if (kernel_execution_count_mode == 2) kernel_update(&info_it->second.path_data[aux_info_size]);
          if (array_of_statuses != MPI_STATUSES_IGNORE){
            array_of_statuses[i].MPI_SOURCE = info_it->second.partner;
            array_of_statuses[i].MPI_TAG = info_it->second.tag;
          }
        }
      }
      complete_comm(*info_it->second.track, info_it->second.path_data, &pt[i], comp_time, (float)0.);
      comp_time=0;
      array_of_requests[i] = MPI_REQUEST_NULL;
    }
  }
  intercept_overhead += MPI_Wtime() - overhead_start_time;
  // If no requests are fake, issue the user communication the safe way.
  if (true_count == count){
    volatile auto last_start_time = MPI_Wtime();
    ret = PMPI_Waitall(count,array_of_requests,array_of_statuses);
    auto waitall_comm_time = MPI_Wtime() - last_start_time;
    overhead_start_time = MPI_Wtime();
    for (int i=0; i<count; i++){
      MPI_Request request = pt[i];
      auto info_it = nonblocking_internal_info.find(request);
      assert(info_it != nonblocking_internal_info.end());
      int rank; MPI_Comm_rank(info_it->second.track->comm,&rank);
      // If receiver or collective, complete the barrier and the path data propagation
      if (!info_it->second.is_sender || info_it->second.partner==-1){
        assert(info_it->second.path_data != nullptr);
        PMPI_Wait(&info_it->second.prop_req, MPI_STATUS_IGNORE);
        if (info_it->second.partner != -1) update_path_profile(info_it->second.path_data,&cp_costs[0],cp_costs_size);
        //if (kernel_execution_count_mode == 2) kernel_update(&info_it->second.path_data[aux_info_size]);
      }
      complete_comm(*info_it->second.track, info_it->second.path_data, &pt[i], comp_time, waitall_comm_time);
      // Although we have to exchange the path data for each request, we do not want to float-count the computation time nor the communicaion time
      comp_time=0; waitall_comm_time=0;
    }
    intercept_overhead += MPI_Wtime() - overhead_start_time;
  }
  else{
    while (true_count>0){
      int indx; MPI_Status status;
      volatile auto start_comm_time = MPI_Wtime();
      int _ret = PMPI_Waitany(count,array_of_requests,&indx,&status);
      auto comm_time = MPI_Wtime() - start_comm_time;
      assert(_ret == MPI_SUCCESS);
      overhead_start_time = MPI_Wtime();
      auto info_it = nonblocking_internal_info.find(pt[indx]);
      assert(info_it != nonblocking_internal_info.end());
      int rank; MPI_Comm_rank(info_it->second.track->comm,&rank);
      // If receiver or collective, complete the barrier and the path data propagation
      if (!info_it->second.is_sender || info_it->second.partner==-1){
        assert(info_it->second.path_data != nullptr);
        PMPI_Wait(&info_it->second.prop_req, MPI_STATUS_IGNORE);
        update_path_profile(info_it->second.path_data,&cp_costs[0],cp_costs_size);
        //if (kernel_execution_count_mode == 2) kernel_update(&info_it->second.path_data[aux_info_size]);
        if (array_of_statuses != MPI_STATUSES_IGNORE){
          array_of_statuses[indx].MPI_SOURCE = info_it->second.partner;
          array_of_statuses[indx].MPI_TAG = info_it->second.tag;
        }
      }
      auto save_r = info_it->first;
      complete_comm(*info_it->second.track, info_it->second.path_data, &save_r, comp_time, comm_time);
      comp_time=0;
      array_of_requests[indx] = MPI_REQUEST_NULL;
      true_count--;
      intercept_overhead += MPI_Wtime() - overhead_start_time;
    }
  }
  return ret;
}
*/

}
}
}
