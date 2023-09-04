#include "comm.h"
#include "../util.h"
#include "../interface.h"

int selectivex_init(int* argc, char*** argv){
  int ret = PMPI_Init(argc,argv);
  selectivex::start(MPI_COMM_WORLD,true);
  return ret;
}

int selectivex_init_thread(int* argc, char*** argv, int required, int* provided){
  assert(required == MPI_THREAD_SINGLE);
  int ret = PMPI_Init_thread(argc,argv,required,provided);
  selectivex::start(MPI_COMM_WORLD,true);
  return ret;
}

int selectivex_finalize(){
  selectivex::stop(MPI_COMM_WORLD,true);
  return PMPI_Finalize();
}

int selectivex_comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm){
  return PMPI_Comm_split(comm,color,key,newcomm);
}

int selectivex_comm_dup(MPI_Comm comm, MPI_Comm* newcomm){
  return PMPI_Comm_dup(comm,newcomm);
}

int selectivex_comm_free(MPI_Comm* comm){
  return PMPI_Comm_free(comm);
}

int selectivex_barrier(MPI_Comm comm){
  assert(selectivex::kernel_map.find("barrier") != selectivex::kernel_map.end());
  // Extract features
  int np;
  MPI_Comm_size(comm, &np);
  std::array<size_t,1> feature_vector = {{np}};
  // always execute a barrier
  selectivex::should_observe("barrier",&feature_vector[0],comm);
  int ret = PMPI_Barrier(comm);
  selectivex::observe("barrier",&feature_vector[0],comm);
  return ret;
}

int selectivex_bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm){
  int ret = MPI_SUCCESS;
/*
  if (selectivex::kernel_map.find("bcast") == selectivex::kernel_map.end()){
    float mpi_confidence_tolerance_threshold = (std::getenv("SELECTIVEX_MPI_CONFIDENCE_TOLERANCE") != NULL ? std::stof(std::getenv("SELECTIVEX_MPI_CONFIDENCE_TOLERANCE")) : 0.05);
    size_t mpi_min_num_executions = (std::getenv("SELECTIVEX_MPI_MIN_NUM_EXECUTIONS") != NULL ? std::stof(std::getenv("SELECTIVEX_MPI_MIN_NUM_EXECUTIONS")) : 3);
    float mpi_min_execution_time = (std::getenv("SELECTIVEX_MPI_MIN_EXECUTION_TIME") != NULL ? std::stof(std::getenv("SELECTIVEX_MPI_MIN_EXECUTION_TIME")) : 1e-3);
    bool mpi_always_synchronize = (std::getenv("SELECTIVEX_MPI_ALWAYS_SYNCHRONIZE") != NULL ? atoi(std::getenv("SELECTIVEX_MPI_ALWAYS_SYNCHRONIZE"))==1 : true);
    selectivex::register_kernel<2,selectivex::internal::NoOpModel>("bcast",[](size_t*,size_t*,size_t*){return false;},true,true,mpi_confidence_tolerance_threshold,mpi_min_num_executions,mpi_min_execution_time,mpi_always_synchronize);
  }
*/
  assert(selectivex::kernel_map.find("bcast") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = word_size * count;
  MPI_Comm_size(comm, &np);
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("bcast",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Bcast(buffer, count, datatype, root, comm);
  }
  else{
    selectivex::predict("bcast",&feature_vector[0]);
  }
  selectivex::observe("bcast",&feature_vector[0],comm);
  return ret;
}

int selectivex_reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("reduce") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = word_size * count;
  MPI_Comm_size(comm, &np);
  std::array<size_t ,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("reduce",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Reduce(sendbuf,recvbuf,count,datatype,op,root,comm);
  }
  else{
    selectivex::predict("reduce",&feature_vector[0]);
  }
  selectivex::observe("reduce",&feature_vector[0],comm);
  return ret;
}

int selectivex_allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("allreduce") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = word_size * count;
  MPI_Comm_size(comm, &np);
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("allreduce",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  }
  else{
    selectivex::predict("allreduce",&feature_vector[0]);
  }
  selectivex::observe("allreduce",&feature_vector[0],comm);
  return ret;
}

int selectivex_gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
           MPI_Datatype recvtype, int root, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("gather") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(sendtype, &word_size);
  size_t nbytes = word_size * sendcount;
  MPI_Comm_size(comm, &np);
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("gather",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
  }
  else{
    selectivex::predict("gather",&feature_vector[0]);
  }
  selectivex::observe("gather",&feature_vector[0],comm);
  return ret;
}

int selectivex_allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
              MPI_Datatype recvtype, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("allgather") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(sendtype, &word_size);
  size_t nbytes = word_size * sendcount;
  MPI_Comm_size(comm, &np);
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("allgather",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  }
  else{
    selectivex::predict("allgather",&feature_vector[0]);
  }
  selectivex::observe("allgather",&feature_vector[0],comm);
  return ret;
}

int selectivex_scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
            MPI_Datatype recvtype, int root, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("scatter") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(sendtype, &word_size);
  size_t nbytes = word_size * sendcount;
  MPI_Comm_size(comm, &np);
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("scatter",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
  }
  else{
    selectivex::predict("scatter",&feature_vector[0]);
  }
  selectivex::observe("scatter",&feature_vector[0],comm);
  return ret;
}

int selectivex_reduce_scatter(const void* sendbuf, void* recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("reduce_scatter") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = 0;
  MPI_Comm_size(comm, &np);
  //TODO: Note that the message size feature as calculated below might be a poor choice and lead to multi-modal execution time distributions.
  for (int i=0; i<np; i++){ nbytes += recvcounts[i]; }
  nbytes *= word_size;
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("reduce_scatter",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
  }
  else{
    selectivex::predict("reduce_scatter",&feature_vector[0]);
  }
  selectivex::observe("reduce_scatter",&feature_vector[0],comm);
  return ret;
}

int selectivex_alltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
             MPI_Datatype recvtype, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("alltoall") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(sendtype, &word_size);
  size_t nbytes = word_size * sendcount;
  MPI_Comm_size(comm, &np);
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("alltoall",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  }
  else{
    selectivex::predict("alltoall",&feature_vector[0]);
  }
  selectivex::observe("alltoall",&feature_vector[0],comm);
  return ret;
}

int selectivex_gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int* recvcounts, const int* displs,
             MPI_Datatype recvtype, int root, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("gatherv") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(recvtype, &word_size);
  size_t nbytes = 0;
  MPI_Comm_size(comm, &np);
  //TODO: Note that the message size feature as calculated below might be a poor choice and lead to multi-modal execution time distributions.
  for (int i=0; i<np; i++){ nbytes += recvcounts[i]; }
  nbytes *= word_size;
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("gatherv",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
  }
  else{
    selectivex::predict("gatherv",&feature_vector[0]);
  }
  selectivex::observe("gatherv",&feature_vector[0],comm);
  return ret;
}

int selectivex_allgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int* recvcounts, const int* displs,
             MPI_Datatype recvtype, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("allgatherv") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(recvtype, &word_size);
  size_t nbytes = 0;
  MPI_Comm_size(comm, &np);
  //TODO: Note that the message size feature as calculated below might be a poor choice and lead to multi-modal execution time distributions.
  for (int i=0; i<np; i++){ nbytes += recvcounts[i]; }
  nbytes *= word_size;
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("allgatherv",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
  }
  else{
    selectivex::predict("allgatherv",&feature_vector[0]);
  }
  selectivex::observe("allgatherv",&feature_vector[0],comm);
  return ret;
}

int selectivex_scatterv(const void* sendbuf, const int* sendcounts, const int* displs, MPI_Datatype sendtype,
              void* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("scatterv") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(recvtype, &word_size);
  size_t nbytes = 0;
  MPI_Comm_size(comm, &np);
  //TODO: Note that the message size feature as calculated below might be a poor choice and lead to multi-modal execution time distributions.
  nbytes = recvcount*word_size;
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("scatterv",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
  }
  else{
    selectivex::predict("scatterv",&feature_vector[0]);
  }
  selectivex::observe("scatterv",&feature_vector[0],comm);
  return ret;
}

int selectivex_alltoallv(const void* sendbuf, const int* sendcounts, const int* sdispls, MPI_Datatype sendtype, void* recvbuf,
               const int* recvcounts, const int* rdispls, MPI_Datatype recvtype, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("alltoallv") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np;
  MPI_Type_size(recvtype, &word_size);
  size_t nbytes = 0;
  MPI_Comm_size(comm, &np);
  //TODO: Note that the message size feature as calculated below might be a poor choice and lead to multi-modal execution time distributions.
  for (int i=0; i<np; i++){ nbytes += recvcounts[i]; }
  nbytes *= word_size;
  std::array<size_t,2> feature_vector = {{(size_t)nbytes,np}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("alltoallv",&feature_vector[0],comm);
  if (schedule_decision){
    ret = PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
  }
  else{
    selectivex::predict("alltoallv",&feature_vector[0]);
  }
  selectivex::observe("alltoallv",&feature_vector[0],comm);
  return ret;
}

int selectivex_sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void* recvbuf, int recvcount,
              MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status* status){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("sendrecv") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np,my_rank;
  MPI_Type_size(sendtype, &word_size);
  size_t nbytes = word_size * sendcount;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &my_rank);
  std::array<size_t,4> feature_vector = {{(size_t)nbytes,np,abs(my_rank-dest),abs(my_rank-source)}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("sendrecv",&feature_vector[0],comm,dest,source);
  if (schedule_decision){
    ret = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status);
  }
  else{
    selectivex::predict("sendrecv",&feature_vector[0]);
  }
  selectivex::observe("sendrecv",&feature_vector[0],comm);
  return ret;
}

int selectivex_sendrecv_replace(void* buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag,
                      MPI_Comm comm, MPI_Status* status){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("sendrecv") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np,my_rank;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = word_size * count;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &my_rank);
  std::array<size_t,4> feature_vector = {{(size_t)nbytes,np,abs(my_rank-dest),abs(my_rank-source)}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("sendrecv",&feature_vector[0],comm,dest,source);
  if (schedule_decision){
    ret = PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, comm, status);
  }
  else{
    selectivex::predict("sendrecv",&feature_vector[0]);
  }
  selectivex::observe("sendrecv",&feature_vector[0],comm);
  return ret;
}

int selectivex_ssend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("send") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np,my_rank;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = word_size * count;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &my_rank);
  std::array<size_t,3> feature_vector = {{(size_t)nbytes,np,abs(my_rank-dest)}};
  // Initiate selective execution
  //TODO: We currently don't require the overhead to be synchronous and consider ssend as send for our purposes.
  bool schedule_decision = selectivex::should_observe("send",&feature_vector[0],comm,dest,true);
  if (schedule_decision){
    ret = PMPI_Ssend(buf, count, datatype, dest, tag, comm);
  }
  else{
    selectivex::predict("send",&feature_vector[0]);
  }
  selectivex::observe("send",&feature_vector[0],comm);
  return ret;
}

int selectivex_bsend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("send") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np,my_rank;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = word_size * count;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &my_rank);
  std::array<size_t,3> feature_vector = {{(size_t)nbytes,np,abs(my_rank-dest)}};
  // Initiate selective execution
  //TODO: We currently treat bsend as a send
  bool schedule_decision = selectivex::should_observe("send",&feature_vector[0],comm,dest,true);
  if (schedule_decision){
    ret = PMPI_Bsend(buf, count, datatype, dest, tag, comm);
  }
  else{
    selectivex::predict("send",&feature_vector[0]);
  }
  selectivex::observe("send",&feature_vector[0],comm);
  return ret;
}

int selectivex_send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("send") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np,my_rank;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = word_size * count;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &my_rank);
  std::array<size_t,3> feature_vector = {{(size_t)nbytes,np,abs(my_rank-dest)}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("send",&feature_vector[0],comm,dest,true);
  if (schedule_decision){
    ret = PMPI_Send(buf, count, datatype, dest, tag, comm);
  }
  else{
    selectivex::predict("send",&feature_vector[0]);
  }
  selectivex::observe("send",&feature_vector[0],comm);
  return ret;
}

int selectivex_recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status){
  assert(source != MPI_ANY_SOURCE);//TODO: Handle this case
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("recv") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np,my_rank;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = word_size * count;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &my_rank);
  std::array<size_t,3> feature_vector = {{(size_t)nbytes,np,abs(my_rank-source)}};
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("recv",&feature_vector[0],comm,source,false);
  if (schedule_decision){
    ret = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
  }
  else{
    selectivex::predict("recv",&feature_vector[0]);
  }
  selectivex::observe("recv",&feature_vector[0],comm);
  return ret;
}

int selectivex_isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request){
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("isend") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np,my_rank;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = word_size * count;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &my_rank);
  std::array<size_t,3> feature_vector = {{(size_t)nbytes,np,abs(my_rank-dest)}};
  double nonblocking_time = 0;
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("isend",&feature_vector[0],comm,dest,true,false);
  if (schedule_decision){
    nonblocking_time = MPI_Wtime();
    ret = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    nonblocking_time = MPI_Wtime()-nonblocking_time;
  } else{
    assert(0);
    selectivex::predict("isend",&feature_vector[0]);
    *request = selectivex::request_id++;
  }
  selectivex::skip_observe(nonblocking_time,true,request);
  return ret;
}

int selectivex_irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request){
  assert(source != MPI_ANY_SOURCE);//TODO: Handle this case
  int ret = MPI_SUCCESS;
  assert(selectivex::kernel_map.find("irecv") != selectivex::kernel_map.end());
  // Extract features
  int word_size,np,my_rank;
  MPI_Type_size(datatype, &word_size);
  size_t nbytes = word_size * count;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &my_rank);
  std::array<size_t,3> feature_vector = {{(size_t)nbytes,np,abs(my_rank-source)}};
  double nonblocking_time = 0;
  // Initiate selective execution
  bool schedule_decision = selectivex::should_observe("irecv",&feature_vector[0],comm,source,false,false);
  if (schedule_decision){
    nonblocking_time = MPI_Wtime();
    ret = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
    nonblocking_time = MPI_Wtime()-nonblocking_time;
  } else{
    assert(0);
    selectivex::predict("irecv",&feature_vector[0]);
    *request = selectivex::request_id++;
  }
  selectivex::skip_observe(nonblocking_time,true,request);
  return ret;
}

int selectivex_ibcast(void* buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request* request){
  int ret = MPI_SUCCESS;
  assert(0);
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[21], curtime, count, datatype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Ibcast(buf, count, datatype, root, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[21], itime, count, datatype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Ibcast(buf, count, datatype, root, comm, request);
  }
*/
  return ret;
}

int selectivex_iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                MPI_Request *request){
  int ret = MPI_SUCCESS;
  assert(0);
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[22], curtime, count, datatype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[22], itime, count, datatype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
  }
  return ret;
*/
}

int selectivex_ireduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm, MPI_Request* request){
  int ret = MPI_SUCCESS;
  assert(0);
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[23], curtime, count, datatype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[23], itime, count, datatype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm, request);
  }
  return ret;
*/
}

int selectivex_igather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
             int root, MPI_Comm comm, MPI_Request* request){
  int ret = MPI_SUCCESS;
  assert(0);
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    int comm_size; MPI_Comm_size(comm, &comm_size);
    int64_t recvbuf_size = std::max((int64_t)sendcount,(int64_t)recvcount) * comm_size;
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[24], curtime, recvbuf_size, sendtype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[24], itime, recvbuf_size, sendtype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
  }
  return ret;
*/
}

int selectivex_igatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcounts[], const int displs[],
              MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request *request){
  int ret = MPI_SUCCESS;
  assert(0);
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    int64_t tot_recv=0; int comm_rank,comm_size; MPI_Comm_rank(comm, &comm_rank); MPI_Comm_size(comm, &comm_size);
    if (comm_rank == root) for (int i=0; i<comm_size; i++){ tot_recv += ((int*)recvcounts)[i]; }
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[25], curtime, std::max((int64_t)sendcount,tot_recv), sendtype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[25], itime, std::max((int64_t)sendcount,tot_recv), sendtype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
     ret = PMPI_Igatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm, request);
  }
  return ret;
*/
}

int selectivex_iallgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
                MPI_Comm comm, MPI_Request* request){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    int comm_size; MPI_Comm_size(comm, &comm_size); int64_t recvbuf_size = std::max((int64_t)sendcount,(int64_t)recvcount) * comm_size;
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[26], curtime, recvbuf_size, sendtype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[26], itime, recvbuf_size, sendtype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  }
  return ret;
*/
}

int selectivex_iallgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcounts[], const int displs[],
                 MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    int64_t tot_recv=0; int comm_size; MPI_Comm_size(comm, &comm_size);
    for (int i=0; i<comm_size; i++){ tot_recv += recvcounts[i]; }
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[27], curtime, std::max((int64_t)sendcount,tot_recv), sendtype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[27], itime, std::max((int64_t)sendcount,tot_recv), sendtype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Iallgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm, request);
  }
  return ret;
*/
}

int selectivex_iscatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
              MPI_Comm comm, MPI_Request* request){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    int comm_size; MPI_Comm_size(comm, &comm_size);
    int64_t sendbuf_size = std::max((int64_t)sendcount,(int64_t)recvcount) * comm_size;
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[28], curtime, sendbuf_size, sendtype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Iscatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[28], itime, sendbuf_size, sendtype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Iscatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
  }
  return ret;
*/
}

int selectivex_iscatterv(const void* sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void* recvbuf, int recvcount,
               MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    int64_t tot_send=0;
    int comm_rank, comm_size; MPI_Comm_rank(comm, &comm_rank); MPI_Comm_size(comm, &comm_size);
    if (comm_rank == root) for (int i=0; i<comm_size; i++){ tot_send += ((int*)sendcounts)[i]; } 
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[29], curtime, std::max(tot_send,(int64_t)recvcount), sendtype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Iscatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[29], itime, std::max(tot_send,(int64_t)recvcount), sendtype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Iscatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm, request);
  }
  return ret;
*/
}

int selectivex_ireduce_scatter(const void* sendbuf, void* recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op,
                     MPI_Comm comm, MPI_Request* request){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    int64_t tot_recv=0;
    int comm_size; MPI_Comm_size(comm, &comm_size);
    for (int i=0; i<comm_size; i++){ tot_recv += recvcounts[i]; }
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[30], curtime, tot_recv, datatype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Ireduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[30], itime, tot_recv, datatype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Ireduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm, request);
  }
  return ret;
*/
}

int selectivex_ialltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
               MPI_Comm comm, MPI_Request* request){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    int comm_size; MPI_Comm_size(comm, &comm_size);
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[31], curtime, std::max((int64_t)sendcount,(int64_t)recvcount)*comm_size, sendtype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[31], itime, std::max((int64_t)sendcount,(int64_t)recvcount)*comm_size, sendtype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, request);
  }
  return ret;
*/
}

int selectivex_ialltoallv(const void* sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void* recvbuf,
                const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_collective){
    volatile auto curtime = MPI_Wtime();
    int64_t tot_send=0, tot_recv=0;
    int comm_size; MPI_Comm_size(comm, &comm_size);
    for (int i=0; i<comm_size; i++){ tot_send += sendcounts[i]; tot_recv += recvcounts[i]; }
    bool schedule_decision = internal::profiler::inspect_comm(*(internal::nonblocking*)internal::list[32], curtime, std::max(tot_send,tot_recv), sendtype, comm);
    if (schedule_decision){
      volatile auto itime = MPI_Wtime();
      ret = PMPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request);
      itime = MPI_Wtime()-itime;
      internal::profiler::initiate_comm(*(internal::nonblocking*)internal::list[32], itime, std::max(tot_send,tot_recv), sendtype, comm, request);
    } else{
      *request = internal::request_id++;
    }
  }
  else{
    ret = PMPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm, request);
  }
  return ret;
*/
}

int selectivex_wait(MPI_Request* request, MPI_Status* status){
  int ret = MPI_SUCCESS;
  selectivex::skip_should_observe(request);
  double start_time = MPI_Wtime();
  ret = PMPI_Wait(request, status);
  selectivex::skip_observe(MPI_Wtime()-start_time,false);
  return ret;
}

int selectivex_waitany(int count, MPI_Request array_of_requests[], int* indx, MPI_Status* status){
  int ret = MPI_SUCCESS;
  //TODO: This protocol will force an ordering which might not match that of the Waitall implementation.
  ret = MPI_Wait(&array_of_requests[0],status); *indx = 0;
  return ret;
}

int selectivex_waitsome(int incount, MPI_Request array_of_requests[], int* outcount, int array_of_indices[], MPI_Status array_of_statuses[]){
  int ret = MPI_SUCCESS;
  //TODO: This protocol will force an ordering which might not match that of the Waitall implementation.
  ret = MPI_Wait(&array_of_requests[0],&array_of_statuses[0]); *outcount = 1; array_of_indices[0] = 0;
  return ret;
}

int selectivex_waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]){
  int ret = MPI_SUCCESS;
  //TODO: This protocol will force an ordering which might not match that of the Waitall implementation.
  for (int i=0; i<count; i++){
    ret = MPI_Wait(&array_of_requests[i],&array_of_statuses[i]);
    assert(ret == MPI_SUCCESS);
  }
  return ret;
}

int selectivex_test(MPI_Request* request, int* flag, MPI_Status* status){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_p2p){
    volatile auto curtime = MPI_Wtime();
    ret = internal::profiler::complete_comm(curtime,request, status,1,flag);
  }
  else{
    ret = PMPI_Test(request, flag, status);
  }
*/
  return ret;
}

int selectivex_testany(int count, MPI_Request array_of_requests[], int* indx, int* flag, MPI_Status* status){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_p2p){
    volatile auto curtime = MPI_Wtime();
    ret = internal::profiler::complete_comm(curtime, count, array_of_requests, indx, status,1,flag);
  }
  else{
    ret = PMPI_Testany(count, array_of_requests, indx, flag, status);
  }
*/
  return ret;
}

int selectivex_testsome(int incount, MPI_Request array_of_requests[], int* outcount, int array_of_indices[], MPI_Status array_of_statuses[]){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_p2p){
    volatile auto curtime = MPI_Wtime();
    ret = internal::profiler::complete_comm(curtime, incount, array_of_requests, outcount, array_of_indices, array_of_statuses,1);
  }
  else{
    ret = PMPI_Testsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
  }
*/
  return ret;
}

int selectivex_testall(int count, MPI_Request array_of_requests[], int* flag, MPI_Status array_of_statuses[]){
  assert(0);
  int ret = MPI_SUCCESS;
/*
  if (internal::mode && internal::profile_p2p){
    volatile auto curtime = MPI_Wtime();
    ret = internal::profiler::complete_comm(curtime,count,array_of_requests,array_of_statuses,1,flag);
  }
  else{
    ret = PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
  }
*/
  return ret;
}
