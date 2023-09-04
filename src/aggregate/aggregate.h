#ifndef SELECTIVEX__AGGREGATE__UTIL_H_
#define SELECTIVEX__AGGREGATE__UTIL_H_

#include <mpi.h>
#include <functional>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <utility>
#include <iomanip>
#include <vector>
#include <stack>
#include <stdint.h>
#include <functional>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_map>
#include <cmath>
#include <assert.h>
#include <climits>

namespace selectivex{
namespace internal{

extern MPI_Datatype comm_kernel_key_type;
extern MPI_Datatype comp_kernel_key_type;

extern struct kernel_batch;
extern struct kernel_propagate;


struct sample_propagation_tree{
 solo_channel* root;
};

struct sample_propagation_forest{
  sample_propagation_forest();
  ~sample_propagation_forest();

  //void generate_span(channel* node, std::vector<std::pair<int,int>>& perm_tuples);
  void insert_node(solo_channel* node);
  void clear_info();
  void fill_ancestors(solo_channel* node, kernel_batch& batch);
  void fill_descendants(solo_channel* node, kernel_batch& batch);

  sample_propagation_tree* tree;
private:
  void delete_tree(solo_channel*& tree_root);
  void clear_tree_info(solo_channel* tree_root);
/*
  void generate_sibling_perm(std::vector<std::pair<int,int>>& static_info, std::vector<std::pair<int,int>>& gen_info, std::vector<std::pair<int,int>>& save_info, int level, bool& valid_siblings);
  void generate_partition_perm(std::vector<std::pair<int,int>>& static_info, std::vector<std::pair<int,int>>& gen_info, int level, bool& valid_partition,
                               int parent_max_span, int parent_min_stride);
*/
  //bool partition_test(channel* parent, int subtree_idx);
  void find_parent(solo_channel* tree_root, solo_channel* tree_node, solo_channel*& parent);
};



// ****************************************************************************************************************************************************
struct channel{
  channel();
  static std::vector<std::pair<int,int>> generate_tuple(std::vector<int>& ranks, int new_comm_size);
  static void contract_tuple(std::vector<std::pair<int,int>>& tuple_list);
  static int enumerate_tuple(channel* node, std::vector<int>& process_list);
  static int duplicate_process_count(std::vector<int>& process_list);
  static int translate_rank(MPI_Comm comm, int rank);
  static bool verify_ancestor_relation(channel* comm1, channel* comm2);
  static bool verify_sibling_relation(channel* comm1, channel* comm2);
  static int span(std::pair<int,int>& id);
  static std::string generate_tuple_string(channel* comm);
 
  int offset;
  int local_hash_tag;
  int global_hash_tag;
  std::vector<std::pair<int,int>> id;
};

struct aggregate_channel : public channel{
  aggregate_channel(std::vector<std::pair<int,int>>& tuple_list, int local_hash, int global_hash, int offset, int channel_size);
  static std::string generate_hash_history(aggregate_channel* comm);

  bool is_final;
  int num_channels;
  std::set<int> channels;
};

struct solo_channel : public channel{
  solo_channel();
  static bool verify_sibling_relation(solo_channel* node, int subtree_idx, std::vector<int>& skip_indices);

  int tag;
  int frequency;
  solo_channel* parent;
  std::vector<std::vector<solo_channel*>> children;
};

extern sample_propagation_forest spf;
extern std::map<MPI_Comm,solo_channel*> comm_channel_map;
extern std::map<int,solo_channel*> p2p_channel_map;
extern std::map<int,aggregate_channel*> aggregate_channel_map;
extern std::map<std::string,std::vector<float>> save_info;

void generate_initial_aggregate();
void generate_aggregate_channels(MPI_Comm oldcomm, MPI_Comm newcomm);
void clear_aggregates();
void comp_state_aggregation(blocking& tracker);
void comm_state_aggregation(blocking& tracker);

//void merge_batches(std::vector<kernel_batch>& batches);


}
}

#endif /*SELECTIVEX__AGGREGATE__UTIL_H_*/
