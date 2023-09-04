#include <cstring>
#include <stdint.h>
#include <limits.h>

#include "util.h"

namespace selectivex{
namespace internal{


MPI_Datatype comm_kernel_key_type;
MPI_Datatype comp_kernel_key_type;
  
  //Included from intercept/comm.cxx. Currently not used.
  //TODO: Likely want to replace. I don't like how it is hard-coded with 7 ints and a float
  comp_kernel_key ex_1;
  MPI_Datatype comp_kernel_key_internal_type[2] = { MPI_INT, MPI_FLOAT };
  int comp_kernel_key_internal_type_block_len[2] = { 7,1 };
  MPI_Aint comp_kernel_key_internal_type_disp[2] = { (char*)&ex_1.tag-(char*)&ex_1, (char*)&ex_1.flops-(char*)&ex_1 };
  PMPI_Type_create_struct(2,comp_kernel_key_internal_type_block_len,comp_kernel_key_internal_type_disp,comp_kernel_key_internal_type,&comp_kernel_key_type);
  PMPI_Type_commit(&comp_kernel_key_type);

  comm_kernel_key ex_2;
  MPI_Datatype comm_kernel_key_internal_type[2] = { MPI_INT, MPI_FLOAT };
  int comm_kernel_key_internal_type_block_len[2] = { 7,1 };
  MPI_Aint comm_kernel_key_internal_type_disp[2] = { (char*)&ex_2.tag-(char*)&ex_2, (char*)&ex_2.msg_size-(char*)&ex_2 };
  PMPI_Type_create_struct(2,comm_kernel_key_internal_type_block_len,comm_kernel_key_internal_type_disp,comm_kernel_key_internal_type,&comm_kernel_key_type);
  PMPI_Type_commit(&comm_kernel_key_type);


sample_propagation_forest spf;
std::map<MPI_Comm,solo_channel*> comm_channel_map;
std::map<int,solo_channel*> p2p_channel_map;
std::map<int,aggregate_channel*> aggregate_channel_map;

// ****************************************************************************************************************************************************

// Reference implementation of get_error_estimte, which is referenced somewhere below.
float get_error_estimate(const comm_kernel_key& key, const kernel_key_id& index){
  auto& active_batches = comm_batch_map[key];
  auto stats = intermediate_stats(index,active_batches);
  return get_confidence_interval(key,stats) / (get_estimate(stats));
}
float get_error_estimate(const comp_kernel_key& key, const kernel_key_id& index){
  auto& active_batches = comp_batch_map[key];
  auto stats = intermediate_stats(index,active_batches);
  return get_confidence_interval(key,stats) / (get_estimate(stats));
}
/*
float get_error_estimate(const comm_kernel_key& key, const kernel_propagate& p){
  return get_confidence_interval(key,p) / (get_estimate(p));
}
float get_error_estimate(const comp_kernel_key& key, const kernel_propagate& p){
  return get_confidence_interval(key,p) / (get_estimate(p));
}
*/


// ****************************************************************************************************************************************************
channel::channel(){
  this->offset = INT_MIN;	// will be updated later
}

std::vector<std::pair<int,int>> channel::generate_tuple(std::vector<int>& ranks, int new_comm_size){
  std::vector<std::pair<int,int>> tuple_list;
  if (new_comm_size<=1){
    tuple_list.emplace_back(new_comm_size,1);
  }
  else{
    int stride = ranks[1]-ranks[0];
    int count = 0;
    int jump=1;
    int extra=0;
    int i=0;
    while (i < ranks.size()-1){
      if ((ranks[i+jump]-ranks[i]) != stride){
        tuple_list.emplace_back(count+extra+1,stride);// node->id.push_back(std::make_pair(count+extra+1,stride));
        stride = ranks[i+1]-ranks[0];
        i += jump;
        if (tuple_list.size()==1){
          jump=count+extra+1;
        }
        else{
          jump = (count+extra+1)*tuple_list[tuple_list.size()-2].first;
        }
        extra=1;
        count = 0;
      } else{
        count++;
        i += jump;
      }
    }
    if (count != 0){
      tuple_list.emplace_back(count+extra+1,stride);//node->id.push_back(std::make_pair(count+extra+1,stride));
    }
  }
  assert(tuple_list.size() >= 1);
  assert(tuple_list.size() <= 2);
  return tuple_list;
}
void channel::contract_tuple(std::vector<std::pair<int,int>>& tuple_list){
  int index=0;
  for (int i=1; i<tuple_list.size(); i++){
    if (tuple_list[index].first*tuple_list[index].second == tuple_list[i].second){
      tuple_list[index].first *= tuple_list[i].first;
      tuple_list[index].second = std::min(tuple_list[index].second,tuple_list[i].second);
    }
    else{
      index++;
    }
  }
  int remove_len = tuple_list.size() - index - 1;
  for (auto i=0; i<remove_len; i++){ tuple_list.pop_back(); }
}
int channel::enumerate_tuple(channel* node, std::vector<int>& process_list){
  int count = 0;
  if (node->id.size()==1){
    int offset = node->offset;
    for (auto i=0; i<node->id[0].first; i++){
      process_list.push_back(offset + i*node->id[0].second);
      count++;
      if (node->id[0].second==0) break;// this might occur if a p2p send/receives with itself
    }
  } else{
    assert(node->id.size()==2);
    bool op = (node->id[0].first*node->id[0].second) < node->id[0].second;
    int max_process;
    if (op){
      max_process = node->offset + channel::span(node->id[0]) + channel::span(node->id[1]);
    } else{
      max_process = node->offset + std::max(channel::span(node->id[0]), channel::span(node->id[1]));
    }
    int offset = node->offset;
    for (auto i=0; i<node->id[1].first; i++){
      for (auto j=0; j<node->id[0].first; j++){
        if (offset + i*node->id[0].second <= max_process){ process_list.push_back(offset + i*node->id[0].second); count++; }
      }
      offset += node->id[1].second;
    }
  }
  return count;
}
int channel::duplicate_process_count(std::vector<int>& process_list){
  int count=0;
  int save = process_list[0];
  for (int i=0; i<process_list.size(); i++){
    if (process_list[i] == save){
      count++;
    } else{
      save = process_list[i];// restart duplicate tracking
    }
  }
  return count;
}
int channel::translate_rank(MPI_Comm comm, int rank){
  // Returns new rank relative to world communicator
  auto node = comm_channel_map[comm];
  int new_rank = node->offset;
  for (auto i=0; i<node->id.size(); i++){
    new_rank += node->id[i].second*(rank%node->id[i].first);
    rank /= node->id[i].first;
  }
  return new_rank;
}
std::string channel::generate_tuple_string(channel* comm){
  std::string str1 = "{ offset = " + std::to_string(comm->offset) + ", ";
  for (auto it : comm->id){
    str1 += " (" + std::to_string(it.first) + "," + std::to_string(it.second) + ")";
  }
  str1+=" }";
  return str1;
}

bool channel::verify_ancestor_relation(channel* comm1, channel* comm2){
/*
  // First check that the parent 'comm2' is not a p2p, regardless of whether 'tree_node' is a subcomm or p2p
  int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  if (comm2->tag >= ((-1)*world_size)) return false;
  //TODO: Commented out is old support for checking if a p2p is a child of a channel. Not sure if still correct.
  // p2p nodes need special machinery for detecting whether its a child of 'node'
  int rank = tree_node->offset - node->offset;//TODO; Fix if tree_node describes a aggregated p2p (rare case, we haven't needed it yet)
  if (rank<0) return false;// corner case
  for (int i=node->id.size()-1; i>=0; i--){
    if (rank >= (node->id[i].first*node->id[i].second)){ return false; }
    if (i>0) { rank %= node->id[i].second; }
  }
  if (rank%node->id[0].second != 0) return false;
  return true;
*/
  if ((comm1->id.size() == 1) && (comm2->id.size() == 1)){
    return ((comm1->offset >= comm2->offset) &&
            (comm1->offset+channel::span(comm1->id[0]) <= comm2->offset+channel::span(comm2->id[0])) &&
            ((comm1->offset - comm2->offset) % comm2->id[0].second == 0) &&
            ((comm1->id[0].second % comm2->id[0].second) == 0));
  }
  // Spill to process list, sort, identify if |comm1| duplicates.
  std::vector<int> process_list;
  int count1 = channel::enumerate_tuple(comm1,process_list);
  int count2 = channel::enumerate_tuple(comm2,process_list);
  std::sort(process_list.begin(),process_list.end());
  assert(process_list.size()>0);
  int count = channel::duplicate_process_count(process_list);
  return (count==count1);
}
bool channel::verify_sibling_relation(channel* comm1, channel* comm2){
  if ((comm1->id.size() == 1) && (comm2->id.size() == 1)){
    int min1 = comm1->offset + span(comm1->id[0]);
    int min2 = comm2->offset + span(comm2->id[0]);
    int _min_ = std::min(min1,min2);
    int max1 = comm1->offset;
    int max2 = comm2->offset;
    int _max_ = std::max(max1,max2);
     // Two special cases -- if stride of one channel is 0, that means that a single intersection point is guaranteed.
    if (comm2->id[0].second == 0 || comm1->id[0].second == 0){ return true; }
    int _lcm_ = lcm(comm1->id[0].second,comm2->id[0].second);
    return _lcm_ > (_min_ - _max_);
  }
  // Spill to process list, sort, identify if 1 duplicate.
  // This is in place of a more specialized routine that iterates over the tuples to identify the pairs that identify as siblings.
  std::vector<int> process_list;
  int count1 = channel::enumerate_tuple(comm1,process_list);
  int count2 = channel::enumerate_tuple(comm2,process_list);
  std::sort(process_list.begin(),process_list.end());
  assert(process_list.size()>0);
  int count = channel::duplicate_process_count(process_list);
  return (count==1);
}
int channel::span(std::pair<int,int>& id){
  return (id.first-1)*id.second;
}

solo_channel::solo_channel(){
  this->tag = 0;	// This member MUST be overwritten immediately after instantiation
  this->frequency=0;
  this->children.push_back(std::vector<solo_channel*>());
}
bool solo_channel::verify_sibling_relation(solo_channel* node, int subtree_idx, std::vector<int>& skip_indices){
  // Perform recursive permutation generation to identify if a permutation of tuples among siblings is valid
  // Return true if node's children are valid siblings
  if (node->children[subtree_idx].size()<=1) return true;
  // Check if all are p2p, all are subcomms, or if there is a mixture.
  int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  int p2p_count=0; int subcomm_count=0;
  for (auto i=0; i<node->children[subtree_idx].size(); i++){
    if (node->children[subtree_idx][i]->tag >= ((-1)*world_size)){ p2p_count++; }
    else { subcomm_count++; }
  }
  if ((subcomm_count>0) && (p2p_count>0)) return false;
  if (subcomm_count==0) return true;// all p2p siblings are fine
/*
  std::vector<std::pair<int,int>> static_info;
  int skip_index=0;
  for (auto i=0; i<node->children[subtree_idx].size(); i++){
    if ((skip_index<skip_indices.size()) && (i==skip_indices[skip_index])){
      skip_index++;
      continue;
    }
    if (node->children[subtree_idx][i]->tag >= ((-1)*world_size)){
      continue;
    }
    for (auto j=0; j<node->children[subtree_idx][i]->id.size(); j++){
      static_info.push_back(node->children[subtree_idx][i]->id[j]);
    }
  }
  std::vector<std::pair<int,int>> gen_info;
  std::vector<std::pair<int,int>> save_info;
  bool valid_siblings=false;
  generate_sibling_perm(static_info,gen_info,save_info,0,valid_siblings);
  if (valid_siblings){
    // Now we must run through the p2p, if any. I don't think we need to check if a p2p is also skipped via 'skip_indices', because if its a child of one of the communicators,
    //   it'll b a child of the generated span
    if (p2p_count>0){
      auto* span_node = new solo_channel;
      generate_span(span_node,save_info);
      for (auto i=0; i<node->children[subtree_idx].size(); i++){
        if (node->children[subtree_idx][i]->tag >= ((-1)*world_size)){
          valid_siblings = channel::verify_ancestor_relation(node->children[subtree_idx][i],span_node);
          if (!valid_siblings){
            delete span_node;
            return false;
          }
        }
      }
      delete span_node;
    }
  }
  return valid_siblings;
*/
  // Compare last node in 'node->children[subtree_idx]' against those that are not skipped
  auto& potential_sibling = node->children[subtree_idx][node->children[subtree_idx].size()-1];
  assert(potential_sibling->id.size() == 1);
  int skip_index=0;
  for (auto i=0; i<node->children[subtree_idx].size()-1; i++){
    if ((skip_index < skip_indices.size()) && (i == skip_indices[skip_index])){
      skip_index++;
      continue;
    }
    assert(node->children[subtree_idx][i]->id.size() == 1);
    int min1 = potential_sibling->offset + span(potential_sibling->id[0]);
    int min2 = node->children[subtree_idx][i]->offset + span(node->children[subtree_idx][i]->id[0]);
    int _min_ = std::min(min1,min2);
    int max1 = potential_sibling->offset;
    int max2 = node->children[subtree_idx][i]->offset;
    int _max_ = std::max(max1,max2);
    int _lcm_ = lcm(potential_sibling->id[0].second,node->children[subtree_idx][i]->id[0].second);
    if (_lcm_ < (_min_-_max_)) return false;
    else{// corner case check
      int count=0;
      if (_max_ == max1){
        if ((_max_ - max2) % node->children[subtree_idx][i]->id[0].second == 0) count++;
      } else{
        if ((_max_ - max1) % potential_sibling->id[0].second == 0) count++;
      }
      if (_min_ == min1){
        if ((min2 - _min_) % node->children[subtree_idx][i]->id[0].second == 0) count++;
      } else{
        if ((min1 - _min_) % potential_sibling->id[0].second == 0) count++;
      }
      // If count == 2, then we have a situation in which the endpoints match, and we must have the lcm be >= _min_-_max_, but > _min_-_max_
      if (count==2){
        if (_lcm_ == (_min_-_max_)) return false;
      }
    }
  }
  return true;
}
aggregate_channel::aggregate_channel(std::vector<std::pair<int,int>>& tuple_list, int local_hash, int global_hash, int offset, int channel_size){
  this->local_hash_tag = local_hash;
  this->global_hash_tag = global_hash;
  this->is_final = false;
  this->num_channels=channel_size;
  this->offset = offset;
  this->id = tuple_list;
}
std::string aggregate_channel::generate_hash_history(aggregate_channel* comm){
  std::string str1 = "{ hashes = ";
  int count=0;
  for (auto it : comm->channels){
    if (count>0) str1 += ",";
    str1 += std::to_string(it);
    count++;
  }
  str1+=" }";
  return str1;
}


void generate_initial_aggregate(){
  int _world_size; MPI_Comm_size(MPI_COMM_WORLD,&_world_size);
  int _world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&_world_rank);
  solo_channel* world_node = new solo_channel();
  world_node->tag = communicator_count++;
  world_node->offset = 0;
  world_node->id.push_back(std::make_pair(_world_size,1));
  world_node->parent=nullptr;
  std::string local_channel_hash_str = ".." + std::to_string(world_node->id[0].first) + "." + std::to_string(world_node->id[0].second);
  std::string global_channel_hash_str = ".." + std::to_string(world_node->id[0].first) + "." + std::to_string(world_node->id[0].second);
  world_node->local_hash_tag = std::hash<std::string>()(local_channel_hash_str);// will avoid any local overlap.
  world_node->global_hash_tag = std::hash<std::string>()(global_channel_hash_str);// will avoid any global overlap.
/*
  sample_propagation_tree* tree = new sample_propagation_tree;
  tree->root = world_node;
  spf.tree = tree;
*/
  comm_channel_map[MPI_COMM_WORLD] = world_node;
  // Always treat 1-communicator channels as trivial aggregate channels.
  aggregate_channel* agg_node = new aggregate_channel(world_node->id,world_node->local_hash_tag,world_node->global_hash_tag,world_node->offset,1);
  agg_node->channels.insert(world_node->local_hash_tag);
  assert(aggregate_channel_map.find(world_node->local_hash_tag) == aggregate_channel_map.end());
  aggregate_channel_map[world_node->local_hash_tag] = agg_node;
  aggregate_channel_map[world_node->local_hash_tag]->is_final=true;
}

void generate_aggregate_channels(MPI_Comm oldcomm, MPI_Comm newcomm){
  int world_comm_size; MPI_Comm_size(MPI_COMM_WORLD,&world_comm_size);
  int old_comm_size; MPI_Comm_size(oldcomm,&old_comm_size);
  int new_comm_size; MPI_Comm_size(newcomm,&new_comm_size);
  int world_comm_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_comm_rank);
  int new_comm_rank; MPI_Comm_rank(newcomm,&new_comm_rank);

  solo_channel* node = new solo_channel();
  std::vector<int> gathered_info(new_comm_size,0);
  PMPI_Allgather(&world_comm_rank,1,MPI_INT,&gathered_info[0],1,MPI_INT,newcomm);

  // Now we detect the "color" (really the stride) via iteration
  // Step 1: subtract out the offset from 0 : assuming that the key arg to comm_split didn't re-shuffle
  //         I can try to use std::min_element vs. writing my own manual loop
  std::sort(gathered_info.begin(),gathered_info.end());
  node->offset = gathered_info[0];
  for (auto i=0; i<gathered_info.size(); i++) { gathered_info[i] -= node->offset; }
  node->id = channel::generate_tuple(gathered_info,new_comm_size);
  node->tag = communicator_count++;
  // Local hash_str will include offset and (size,stride) of each tuple
  // Global hash_str will include (size,stride) of each tuple -> not this may be changed (break example would be diagonal+row)
  // Note: I want to incorporate 'node->offset' into local_channel_hash_str', yet the issue is that when I go to iterate over aggregates,
  //   there is no guarantee that the global hash tags will be in same sorted order as local_hash_tag.
  std::string local_channel_hash_str = "";//std::to_string(node->offset);
  std::string global_channel_hash_str = "";
  for (auto i=0; i<node->id.size(); i++){
    local_channel_hash_str += ".." + std::to_string(node->id[i].first) + "." + std::to_string(node->id[i].second);
    global_channel_hash_str += ".." + std::to_string(node->id[i].first) + "." + std::to_string(node->id[i].second);
  }
  node->local_hash_tag = std::hash<std::string>()(local_channel_hash_str);// will avoid any local overlap.
  node->global_hash_tag = std::hash<std::string>()(global_channel_hash_str);// will avoid any global overlap.
  //spf.insert_node(node);// This call will just fill in SPT via node's parent/children members, and the members of related channels
  comm_channel_map[newcomm] = node;

  if (new_comm_size<=1){
    // Just don't even create an aggregate for size-1 communicators.
    return;
  }

  // Recursively build up other legal aggregate channels that include 'node'
  std::vector<int> local_hash_array;
  std::vector<aggregate_channel*> new_aggregate_channels;
  int max_sibling_node_size=0;
  std::vector<int> save_max_indices;
  // Check if 'node' is a sibling of all existing aggregates already formed. Note that we do not include p2p aggregates, nor p2p+comm aggregates.
  // Note this loop assumes that the local_hash_tags of each aggregate across new_comm are in the same sorted order (hence the assert below)

  for (auto it : aggregate_channel_map){
    // 0. Check that each process in newcomm is processing the same aggregate.
    int verify_global_agg_hash1,verify_global_agg_hash2;
    PMPI_Allreduce(&it.second->global_hash_tag,&verify_global_agg_hash1,1,MPI_INT,MPI_MIN,newcomm);
    PMPI_Allreduce(&it.second->global_hash_tag,&verify_global_agg_hash2,1,MPI_INT,MPI_MAX,newcomm);
    // if (verify_global_agg_hash1 != verify_global_agg_hash2) std::cout << "Verify - " << verify_global_agg_hash1 << " " << " " << verify_global_agg_hash2 << it.second->global_hash_tag << std::endl; 
    assert(verify_global_agg_hash1 == verify_global_agg_hash2);
    // 1. Check if 'node' is a child of 'aggregate'
    bool is_child_1 = channel::verify_ancestor_relation(it.second,node);
    // 2. Check if 'aggregate' is a child of 'node'
    bool is_child_2 = channel::verify_ancestor_relation(node,it.second);
    // 3. Check if 'node'+'aggregate' form a sibling
    bool is_sibling = channel::verify_sibling_relation(it.second,node);
    if (is_sibling && !is_child_1 && !is_child_2){
      // If current aggregate forms a larger one with 'node', reset its 'is_final' member to be false, and always set a new aggregate's 'is_final' member to true
      it.second->is_final = false;
      int new_local_hash_tag = it.second->local_hash_tag ^ node->local_hash_tag;
      int new_global_hash_tag = it.second->global_hash_tag ^ node->global_hash_tag;
      auto new_aggregate_channel = new aggregate_channel(it.second->id,new_local_hash_tag,new_global_hash_tag,0,it.second->num_channels+1);// '0' gets updated below
      // Set the hashes of each communicator.
      new_aggregate_channel->channels.insert(node->local_hash_tag);
      for (auto it_2 : it.second->channels){
        new_aggregate_channel->channels.insert(it_2);
      }
      // Communicate to attain the minimum offset of all process in newcomm's aggregate channel.
      PMPI_Allgather(&it.second->offset,1,MPI_INT,&gathered_info[0],1,MPI_INT,newcomm);
      std::sort(gathered_info.begin(),gathered_info.end());
      new_aggregate_channel->offset = gathered_info[0];
      assert(new_aggregate_channel->offset <= it.second->offset);
      for (auto i=0; i<gathered_info.size(); i++) { gathered_info[i] -= new_aggregate_channel->offset; }
      auto tuple_list = channel::generate_tuple(gathered_info,new_comm_size);
      // Generate IR for new aggregate by replacing newcomm's tuple with that of the offsets of its distinct aggregates.
      for (auto it_2 : tuple_list){
        new_aggregate_channel->id.push_back(it_2);
      }
      std::sort(new_aggregate_channel->id.begin(),new_aggregate_channel->id.end(),[](const std::pair<int,int>& p1, const std::pair<int,int>& p2){return p1.second < p2.second;});
      channel::contract_tuple(new_aggregate_channel->id);
      new_aggregate_channels.push_back(new_aggregate_channel);
      local_hash_array.push_back(new_local_hash_tag);
      if (new_aggregate_channels[new_aggregate_channels.size()-1]->num_channels > max_sibling_node_size){
        max_sibling_node_size = new_aggregate_channels[new_aggregate_channels.size()-1]->num_channels;
        save_max_indices.clear();
        save_max_indices.push_back(new_aggregate_channels.size()-1);
      }
      else if (new_aggregate_channels[new_aggregate_channels.size()-1]->num_channels == max_sibling_node_size){
        save_max_indices.push_back(new_aggregate_channels.size()-1);
      }
    }
  }
  // Populate the aggregate_channel_map with the saved pointers that were created in the loop above.
  int index_window=0;
  for (auto i=0; i<new_aggregate_channels.size(); i++){
    // Update is_final to true iff its the largest subset size that includes 'node' (or if there are multiple)
    if ((index_window < save_max_indices.size()) && (save_max_indices[index_window]==i)){ 
      new_aggregate_channels[i]->is_final=true;
      index_window++;
    }
    // assert(aggregate_channel_map.find(new_aggregate_channels[i]->local_hash_tag) == aggregate_channel_map.end());
    if (aggregate_channel_map.find(new_aggregate_channels[i]->local_hash_tag) == aggregate_channel_map.end()){
      aggregate_channel_map[new_aggregate_channels[i]->local_hash_tag] = new_aggregate_channels[i];
    }
  }

  // Verify that the aggregates are build with the same hashes
  int local_sibling_size = local_hash_array.size();
  // Always treat 1-communicator channels as trivial aggregate channels.
  aggregate_channel* agg_node = new aggregate_channel(node->id,node->local_hash_tag,node->global_hash_tag,node->offset,1);
  agg_node->channels.insert(node->local_hash_tag);
  // assert(aggregate_channel_map.find(node->local_hash_tag) == aggregate_channel_map.end());
  if (aggregate_channel_map.find(node->local_hash_tag) == aggregate_channel_map.end()){
    aggregate_channel_map[node->local_hash_tag] = agg_node;
    if (local_sibling_size==0){// Only if 'node' exists as the smallest trivial aggregate should it be considered final. Think of 'node==world' of the very first registered channel
      aggregate_channel_map[node->local_hash_tag]->is_final=true;
    } else{
    }
  }
}

void clear_aggregates(){
  for (auto it : comm_channel_map) delete it.second;
  for (auto it : p2p_channel_map) delete it.second;
  for (auto it : aggregate_channel_map) delete it.second;
  comm_channel_map.clear();
  p2p_channel_map.clear();
  aggregate_channel_map.clear();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
void path::comp_state_aggregation(blocking& tracker){
  int size; MPI_Comm_size(tracker.comm,&size);
  int rank; MPI_Comm_rank(tracker.comm,&rank);
  std::vector<kernel_propagate> foreign_active_kernels;
  std::map<comp_kernel_key,kernel_propagate> save_comp_kernels;

  // First save the kernels we want to contribute to the aggregation (because they are steady)
  for (auto& it : tracker.save_comp_key){
    save_comp_kernels[it] = active_kernels[comp_kernel_map[it].val_index];
  }

  size_t active_size = size;
  size_t active_rank = rank;
  size_t active_mult = 1;
  while (active_size>1){
    if (active_rank % 2 == 1){
      // Fill-in the associated comp kernel
      tracker.save_comp_key.clear();
      foreign_active_kernels.clear();
      for (auto& it : save_comp_kernels){
        tracker.save_comp_key.push_back(it.first);
        foreign_active_kernels.push_back(it.second);
      }

      int partner = (active_rank-1)*active_mult;
      int size_array[2] = {tracker.save_comp_key.size(),tracker.save_comp_key.size()};
      // Send sizes before true message so that receiver can be aware of the array sizes for subsequent communication
      PMPI_Send(&size_array[0],2,MPI_INT,partner,internal_tag,tracker.comm);
      // Send active kernels with keys
      PMPI_Send(&tracker.save_comp_key[0],size_array[0],comp_kernel_key_type,partner,internal_tag2,tracker.comm);
      PMPI_Send(&foreign_active_kernels[0],size_array[1],kernel_type,partner,internal_tag2,tracker.comm);
      break;// Incredibely important. Senders must not update {active_size,active_rank,active_mult}
    }
    else if ((active_rank % 2 == 0) && (active_rank < (active_size-1))){
      int partner = (active_rank+1)*active_mult;
      int size_array[2] = {0,0};
      // Recv sizes of arrays to create buffers for subsequent communication
      PMPI_Recv(&size_array[0],2,MPI_INT,partner,internal_tag,tracker.comm,MPI_STATUS_IGNORE);
      // Recv partner's active kernels with keys
      tracker.save_comp_key.resize(size_array[0]);
      foreign_active_kernels.resize(size_array[1]);
      PMPI_Recv(&tracker.save_comp_key[0],size_array[0],comp_kernel_key_type,partner,internal_tag2,tracker.comm,MPI_STATUS_IGNORE);
      PMPI_Recv(&foreign_active_kernels[0],size_array[1],kernel_type,partner,internal_tag2,tracker.comm,MPI_STATUS_IGNORE);
      // Iterate over all active kernels and simply perform an AND operation on whether a kernel is in steady state.
      //   If just one is active across the world communicator, the kernel must remain active.
      //   If kernel does not exist among the sent kernels, it does not count as active. The logical operation is a trivial (AND 1)
      for (auto i=0; i<tracker.save_comp_key.size(); i++){
        auto& key = tracker.save_comp_key[i];
        if (save_comp_kernels.find(key) != save_comp_kernels.end()){
          auto ci_local = get_error_estimate(key,save_comp_kernels[key]);
          auto ci_foreign = get_error_estimate(key,foreign_active_kernels[i]);
          if (ci_foreign < ci_local){
            save_comp_kernels[key] = foreign_active_kernels[i];
          }
        } else{
          save_comp_kernels[key] = foreign_active_kernels[i];
        }
      }
    }
    active_size = active_size/2 + active_size%2;
    active_rank /= 2;
    active_mult *= 2;
  }
  // Broadcast final exchanged kernel statistics
  if (rank==0){
    tracker.save_comp_key.clear();
    foreign_active_kernels.clear();
    for (auto& it : save_comp_kernels){
      tracker.save_comp_key.push_back(it.first);
      foreign_active_kernels.push_back(it.second);
    }
  }
  int size_array[2] = {rank==0 ? tracker.save_comp_key.size() : 0,
                       rank==0 ? tracker.save_comp_key.size() : 0};
  PMPI_Bcast(&size_array[0],2,MPI_INT,0,tracker.comm);
  if (rank != 0){
    tracker.save_comp_key.resize(size_array[0]);
    foreign_active_kernels.resize(size_array[1]);
  }
  PMPI_Bcast(&tracker.save_comp_key[0],size_array[0],comp_kernel_key_type,0,tracker.comm);
  PMPI_Bcast(&foreign_active_kernels[0],size_array[1],kernel_type,0,tracker.comm);
  for (auto i=0; i<tracker.save_comp_key.size(); i++){
    auto& key = tracker.save_comp_key[i];
    if (comp_kernel_map.find(key) != comp_kernel_map.end()){
      active_kernels[comp_kernel_map[key].val_index].hash_id = foreign_active_kernels[i].hash_id;
    } else{
      // Add new entry.
      active_comp_kernel_keys.push_back(key);
      active_kernels.emplace_back(foreign_active_kernels[i]);
      comp_kernel_map[key] = kernel_key_id(true,active_comp_kernel_keys.size()-1,active_kernels.size()-1,false);
    }
    set_kernel_state(comp_kernel_map[key],false);
    if (propagate_kernel_execution_state==1){
      set_kernel_state_global(comp_kernel_map[key],false);
    } else if (propagate_kernel_execution_state==2){
      if (aggregate_channel_map[comm_channel_map[tracker.comm]->global_hash_tag]->is_final){
        active_kernels[comp_kernel_map[key].val_index].hash_id = aggregate_channel_map[comm_channel_map[tracker.comm]->global_hash_tag]->global_hash_tag;
        active_kernels[comp_kernel_map[key].val_index].registered_channels.clear();
        active_kernels[comp_kernel_map[key].val_index].registered_channels.insert(comm_channel_map[tracker.comm]);// Add the solo channel, not the aggregate
        assert(aggregate_channel_map.find(active_kernels[comp_kernel_map[key].val_index].hash_id) != aggregate_channel_map.end());
        set_kernel_state_global(comp_kernel_map[key],false);
      }
      else{
        active_kernels[comp_kernel_map[key].val_index].hash_id ^= aggregate_channel_map[comm_channel_map[tracker.comm]->global_hash_tag]->global_hash_tag;
        active_kernels[comp_kernel_map[key].val_index].registered_channels.insert(comm_channel_map[tracker.comm]);// Add the solo channel, not the aggregate
        assert(aggregate_channel_map.find(active_kernels[comp_kernel_map[key].val_index].hash_id) != aggregate_channel_map.end());
        if (aggregate_channel_map[active_kernels[comp_kernel_map[key].val_index].hash_id]->is_final){
          set_kernel_state_global(comp_kernel_map[key],false);
        }
      }
    }
  }
  tracker.save_comp_key.clear();
}

void path::comm_state_aggregation(blocking& tracker){
  int size; MPI_Comm_size(tracker.comm,&size);
  int rank; MPI_Comm_rank(tracker.comm,&rank);
  std::vector<kernel_propagate> foreign_active_kernels;
  std::map<comm_kernel_key,kernel_propagate> save_comm_kernels;

  // First save the kernels we want to contribute to the aggregation (because they are steady)
  for (auto& it : tracker.save_comm_key){
    save_comm_kernels[it] = active_kernels[comm_kernel_map[it].val_index];
  }

  size_t active_size = size;
  size_t active_rank = rank;
  size_t active_mult = 1;
  while (active_size>1){
    if (active_rank % 2 == 1){
      // Fill-in the associated comm kernel
      tracker.save_comm_key.clear();
      foreign_active_kernels.clear();
      for (auto& it : save_comm_kernels){
        tracker.save_comm_key.push_back(it.first);
        foreign_active_kernels.push_back(it.second);
      }

      int partner = (active_rank-1)*active_mult;
      int size_array[2] = {tracker.save_comm_key.size(),tracker.save_comm_key.size()};
      // Send sizes before true message so that receiver can be aware of the array sizes for subsequent communication
      PMPI_Send(&size_array[0],2,MPI_INT,partner,internal_tag,tracker.comm);
      // Send active kernels with keys
      PMPI_Send(&tracker.save_comm_key[0],size_array[0],comm_kernel_key_type,partner,internal_tag2,tracker.comm);
      PMPI_Send(&foreign_active_kernels[0],size_array[1],kernel_type,partner,internal_tag2,tracker.comm);
      break;// Incredibely important. Senders must not update {active_size,active_rank,active_mult}
    }
    else if ((active_rank % 2 == 0) && (active_rank < (active_size-1))){
      int partner = (active_rank+1)*active_mult;
      int size_array[2] = {0,0};
      // Recv sizes of arrays to create buffers for subsequent communication
      PMPI_Recv(&size_array[0],2,MPI_INT,partner,internal_tag,tracker.comm,MPI_STATUS_IGNORE);
      // Recv partner's active kernels with keys
      tracker.save_comm_key.resize(size_array[0]);
      foreign_active_kernels.resize(size_array[1]);
      PMPI_Recv(&tracker.save_comm_key[0],size_array[0],comm_kernel_key_type,partner,internal_tag2,tracker.comm,MPI_STATUS_IGNORE);
      PMPI_Recv(&foreign_active_kernels[0],size_array[1],kernel_type,partner,internal_tag2,tracker.comm,MPI_STATUS_IGNORE);
      // Iterate over all active kernels and simply perform an AND operation on whether a kernel is in steady state.
      //   If just one is active across the world communicator, the kernel must remain active.
      //   If kernel does not exist among the sent kernels, it does not count as active. The logical operation is a trivial (AND 1)
      for (auto i=0; i<tracker.save_comm_key.size(); i++){
        auto& key = tracker.save_comm_key[i];
        if (save_comm_kernels.find(key) != save_comm_kernels.end()){
          auto ci_local = get_error_estimate(key,save_comm_kernels[key]);
          auto ci_foreign = get_error_estimate(key,foreign_active_kernels[i]);
          if (ci_foreign < ci_local){
            save_comm_kernels[key] = foreign_active_kernels[i];
          }
        } else{
          save_comm_kernels[key] = foreign_active_kernels[i];
        }
      }
    }
    active_size = active_size/2 + active_size%2;
    active_rank /= 2;
    active_mult *= 2;
  }
  // Broadcast final exchanged kernel statistics
  if (rank==0){
    tracker.save_comm_key.clear();
    foreign_active_kernels.clear();
    for (auto& it : save_comm_kernels){
      tracker.save_comm_key.push_back(it.first);
      foreign_active_kernels.push_back(it.second);
    }
  }
  int size_array[2] = {rank==0 ? tracker.save_comm_key.size() : 0,
                       rank==0 ? tracker.save_comm_key.size() : 0};
  PMPI_Bcast(&size_array[0],2,MPI_INT,0,tracker.comm);
  if (rank != 0){
    tracker.save_comm_key.resize(size_array[0]);
    if (comm_kernel_map.find(key) != comm_kernel_map.end()){
      active_kernels[comm_kernel_map[key].val_index].hash_id = foreign_active_kernels[i].hash_id;
    } else{
      // Add new entry.
      active_comm_kernel_keys.push_back(key);
      active_kernels.emplace_back(foreign_active_kernels[i]);
      comm_kernel_map[key] = kernel_key_id(true,active_comm_kernel_keys.size()-1,active_kernels.size()-1,false);
    }
    set_kernel_state(comm_kernel_map[key],false);
    if (propagate_kernel_execution_state==1){
      set_kernel_state_global(comm_kernel_map[key],false);
    } else if (propagate_kernel_execution_state==2){
      if (aggregate_channel_map[comm_channel_map[tracker.comm]->global_hash_tag]->is_final){
        active_kernels[comm_kernel_map[key].val_index].hash_id = aggregate_channel_map[comm_channel_map[tracker.comm]->global_hash_tag]->global_hash_tag;
        active_kernels[comm_kernel_map[key].val_index].registered_channels.clear();
        active_kernels[comm_kernel_map[key].val_index].registered_channels.insert(comm_channel_map[tracker.comm]);// Add the solo channel, not the aggregate
        assert(aggregate_channel_map.find(active_kernels[comm_kernel_map[key].val_index].hash_id) != aggregate_channel_map.end());
        set_kernel_state_global(comm_kernel_map[key],false);
      }
      else{
        active_kernels[comm_kernel_map[key].val_index].hash_id ^= aggregate_channel_map[comm_channel_map[tracker.comm]->global_hash_tag]->global_hash_tag;
        active_kernels[comm_kernel_map[key].val_index].registered_channels.insert(comm_channel_map[tracker.comm]);// Add the solo channel, not the aggregate
        assert(aggregate_channel_map.find(active_kernels[comm_kernel_map[key].val_index].hash_id) != aggregate_channel_map.end());
        if (aggregate_channel_map[active_kernels[comm_kernel_map[key].val_index].hash_id]->is_final){
          set_kernel_state_global(comm_kernel_map[key],false);
        }
      }
    }
  }
  tracker.save_comm_key.clear();
}
*/


// ****************************************************************************************************************************************************

/*
void sample_propagation_forest::generate_span(channel* node, std::vector<std::pair<int,int>>& perm_tuples){
  // Assumed that perm_tuples define a permutation of channels that are communicator siblings
  //   Thus, 'perm_tuples' will be in sorted order of stride
  // Only need to fill out the 'id' member
  node->id.push_back(perm_tuples[0]);
  int index=0;
  for (int i=1; i<perm_tuples.size(); i++){
    if (node->id[index].first*node->id[index].second == perm_tuples[i].second){
      node->id[index].first *= perm_tuples[i].first;
      node->id[index].second = std::min(node->id[index].second,perm_tuples[i].second);
    }
    else if (node->id[index].first*node->id[index].second < perm_tuples[i].second){
      index++;
      node->id.push_back(perm_tuples[i]);
    }
    else { assert(0); }
  }
}
*/
/*
void sample_propagation_forest::generate_sibling_perm(std::vector<std::pair<int,int>>& static_info, std::vector<std::pair<int,int>>& gen_info,
                                                      std::vector<std::pair<int,int>>& save_info, int level, bool& valid_siblings){
  // static_info will shrink as tuples are transfered into gen_info. That means when 
  if (static_info.size()==0){
    save_info = gen_info;
    valid_siblings=true;
    return;
  }
  // At position 'level' in permutation, lets try all remaining possibilities via iterating over what remains of static_info
  for (auto i=0; i<static_info.size(); i++){
    // check if valid BEFORE recursing, except if at first level (no pruning at that level, all possibilities still valid)
    if ((level==0) || (static_info[i].second == (gen_info[level-1].first*gen_info[level-1].second))){
      gen_info.push_back(static_info[i]);
      if (i==static_info.size()-1){
        static_info.pop_back();
        this->generate_sibling_perm(static_info,gen_info,save_info,level+1,valid_siblings);
        static_info.push_back(gen_info[gen_info.size()-1]);
      } else{
        // swap with last entry and then pop
        auto temp = static_info[static_info.size()-1];
        static_info[static_info.size()-1] = static_info[i];
        static_info[i] = temp;
        static_info.pop_back();
        this->generate_sibling_perm(static_info,gen_info,save_info,level+1,valid_siblings);
        static_info.push_back(temp);
        static_info[i] = gen_info[gen_info.size()-1];
      }
      gen_info.pop_back();
    }
  }
}
*/
/*
void sample_propagation_forest::generate_partition_perm(std::vector<std::pair<int,int>>& static_info, std::vector<std::pair<int,int>>& gen_info, int level,
                                                        bool& valid_partition, int parent_max_span, int parent_min_stride){
  // static_info will shrink as tuples are transfered into gen_info. That means when 
  if (static_info.size()==0){
    if ((level>0) && (gen_info[0].second == parent_min_stride) && (gen_info[level-1].first*gen_info[level-1].second == parent_max_span)){ valid_partition=true; }
    return;
  }
  // At position 'level' in permutation, lets try all remaining possibilities via iterating over what remains of static_info
  int static_info_size=static_info.size();
  for (auto i=0; i<static_info_size; i++){
    // check if valid BEFORE recursing, except if at first level (no pruning at that level, all possibilities still valid)
    if ((level==0) || (static_info[i].second == (gen_info[level-1].first*gen_info[level-1].second))){
      //if ((level==0) && static_info[i].second!= 1) continue;// constrain initial stride in permutation
      gen_info.push_back(static_info[i]);
      if (i==static_info.size()-1){
        static_info.pop_back();
        this->generate_partition_perm(static_info,gen_info,level+1,valid_partition,parent_max_span,parent_min_stride);
        static_info.push_back(gen_info[gen_info.size()-1]);
      } else{
        // swap with last entry and then pop
        auto temp = static_info[static_info.size()-1];
        static_info[static_info.size()-1] = static_info[i];
        static_info[i] = temp;
        static_info.pop_back();
        this->generate_partition_perm(static_info,gen_info,level+1,valid_partition,parent_max_span,parent_min_stride);
        static_info.push_back(temp);
        static_info[i] = gen_info[gen_info.size()-1];
      }
      gen_info.pop_back();
    }
  }
}
*/
/*
bool sample_propagation_forest::partition_test(channel* parent, int subtree_idx){
  // Perform recursive permutation generation to identify if a permutation of tuples among siblings is valid
  // Return true if parent's children are valid siblings
  int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);
  std::vector<std::pair<int,int>> static_info;
  for (auto i=0; i<parent->children[subtree_idx].size(); i++){
    for (auto j=0; j<parent->children[subtree_idx][i]->id.size(); j++){
      if (parent->children[subtree_idx][i]->tag < ((-1)*world_size)){
        static_info.push_back(parent->children[subtree_idx][i]->id[j]);
      }
    }
  }
  std::vector<std::pair<int,int>> gen_info;
  bool valid_partition=false;
  auto parent_max_span = parent->id[parent->id.size()-1].first * parent->id[parent->id.size()-1].second;
  auto parent_min_stride = parent->id[0].second;
  generate_partition_perm(static_info,gen_info,0,valid_partition,parent_max_span,parent_min_stride);
  return valid_partition;
}
*/
void sample_propagation_forest::find_parent(solo_channel* tree_root, solo_channel* tree_node, solo_channel*& parent){
  if (tree_root==nullptr) return;
  for (auto i=0; i<tree_root->children.size(); i++){
    for (auto j=0; j<tree_root->children[i].size(); j++){
      this->find_parent(tree_root->children[i][j],tree_node,parent);// Cannot be nullptrs. Nullptr children mean the children member is empty
    }
  }
  if ((parent==nullptr) && (channel::verify_ancestor_relation(tree_node,tree_root))){
    parent = tree_root;
  }
  return;
}
void sample_propagation_forest::fill_ancestors(solo_channel* node, kernel_batch& batch){
/*
  if (node==nullptr) return;
  batch.closed_channels.insert(node);
  this->fill_ancestors(node->parent,batch);
*/
}
void sample_propagation_forest::fill_descendants(solo_channel* node, kernel_batch& batch){
/*
  if (node==nullptr) return;
  batch.closed_channels.insert(node);
  for (auto i=0; i<node->children.size(); i++){
    for (auto j=0; j<node->children[i].size(); j++){
      this->fill_descendants(node->children[i][j],batch);
    }
  }
*/
}
void sample_propagation_forest::clear_tree_info(solo_channel* tree_root){
  if (tree_root==nullptr) return;
  for (auto i=0; i<tree_root->children.size(); i++){
    for (auto j=0; j<tree_root->children[i].size(); j++){
      this->clear_tree_info(tree_root->children[i][j]);// Cannot be nullptrs. Nullptr children mean the children member is empty
    }
  }
  tree_root->frequency=0;
  return;
}
void sample_propagation_forest::delete_tree(solo_channel*& tree_root){
  if (tree_root==nullptr) return;
  for (auto i=0; i<tree_root->children.size(); i++){
    for (auto j=0; j<tree_root->children[i].size(); j++){
      this->delete_tree(tree_root->children[i][j]);// Cannot be nullptrs. Nullptr children mean the children member is empty
    }
  }
  free(tree_root);
  tree_root=nullptr;
  return;
}
sample_propagation_forest::sample_propagation_forest(){ this->tree=nullptr; }
sample_propagation_forest::~sample_propagation_forest(){
  if (this->tree == nullptr) return;
  this->delete_tree(this->tree->root);
  free(this->tree); this->tree = nullptr;
}
void sample_propagation_forest::clear_info(){
  this->clear_tree_info(this->tree->root);
}
void sample_propagation_forest::insert_node(solo_channel* node){
  // Fill in parent and children, and iterate over all trees of course.
  // Post-order traversal
  // Follow rules from paper to deduce first whether node can be a child of the current parent.
  assert(node != nullptr);
  bool is_comm = !(node->id.size()==1 && node->id[0].second==0);
  solo_channel* parent = nullptr;
  //TODO: I assume here that we care about the first SPT in the SPF. Figure out how to fix this later
  this->find_parent(this->tree->root,node,parent);
  node->parent = parent;
  assert(parent!=nullptr);
 
  // Try adding 'node' to each SPT. If none fit, append parent's children array and add it there, signifying a new tree, rooted at 'parent'
  bool valid_parent = false;
  int save_tree_idx=-1;
  for (auto i=0; i<parent->children.size(); i++){
    parent->children[i].push_back(node);
    std::vector<int> sibling_to_child_indices;
    for (auto j=0; j<parent->children[i].size()-1; j++){
      if (channel::verify_ancestor_relation(parent->children[i][j],node)){
        sibling_to_child_indices.push_back(j);
      }
    }
    bool sibling_decision = solo_channel::verify_sibling_relation(parent,i,sibling_to_child_indices);
    if (sibling_decision){
      save_tree_idx=i;
      valid_parent=true;
      for (auto j=0; j<sibling_to_child_indices.size(); j++){
        node->children[0].push_back(parent->children[i][sibling_to_child_indices[j]]);
        parent->children[i][sibling_to_child_indices[j]]->parent=node;
      }
      int skip_index=0;
      int save_index=0;
      for (auto j=0; j<parent->children[i].size()-1; j++){
        if ((skip_index<sibling_to_child_indices.size()) && (j==sibling_to_child_indices[skip_index])){
          skip_index++;
        } else{
          parent->children[i][save_index] = parent->children[i][j];
          save_index++;
        }
      }
      parent->children[i][save_index] = parent->children[i][parent->children[i].size()-1];
      save_index++;
      for (auto j=parent->children[i].size()-save_index; j>0; j--){
        parent->children[i].pop_back();
      }
      break;
    } else{
      parent->children[i].pop_back();
    }
  }
  if (!valid_parent){
    parent->children.push_back(std::vector<solo_channel*>());
    parent->children[parent->children.size()-1].push_back(node);
  }
}

/*
void merge_batches(std::vector<kernel_batch>& batches){
  if (batches.size() == 0) return;
  // At first, I wanted to leverage the 'is_final' info from the aggregate (which can be obtained from a batch's hash_id
  //   However, I think I don't need to do this. I can just keep that state within the batch and accumulate in there.
  // Iterate over the entires to merge any two batches with the same state.
  std::sort(batches.begin(),batches.end(),[](const kernel_batch& p1, const kernel_batch& p2){return p1.hash_id < p2.hash_id;});
   // Assumption: no more than 2 batches can have the same state.
  int start_index = 0;
  for (auto i=1; i<batches.size(); i++){
    if (batches[i].hash_id == batches[start_index].hash_id){
      // Merge batches[i] into batches[start_index]
      update_model(batches[start_index],batches[i]);
      batches[start_index].num_local_schedules += batches[i].num_local_schedules;
      batches[start_index].num_local_scheduled_units += batches[i].num_local_scheduled_units;
      batches[start_index].total_local_exec_time += batches[i].total_local_exec_time;
    } else{
      batches[++start_index] = batches[i];
    }
  }
  for (int i=batches.size()-start_index-1; i>0; i--){
    batches.pop_back();
  }
}
*/


}
}
