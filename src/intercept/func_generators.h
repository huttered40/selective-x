#ifndef SELECTIVEX__INTERCEPT__FUNCGENERATORS_H_
#define SELECTIVEX__INTERCEPT__FUNCGENERATORS_H_

#include <tuple>
#include <cstring>
#include "../util.h"
#include "../interface.h"

namespace selectivex{

template<size_t... indx>
class IndexPack{};

// Necessary matrix resets before invoking LAPACK routines (which inspect input matrices before performing operation)
class reset_matrix{
public:
  static void invoke(){}

  template<typename TupleType, typename... TupleTypes>
  static void invoke(TupleType&& t, TupleTypes... ts){
    auto matrix = std::get<0>(t);
    int m = std::get<1>(t);
    int n = std::get<2>(t);
    int lda = std::get<3>(t);
    auto reset_val = std::get<4>(t);
    if (m==lda) memset(matrix,reset_val,m*n*sizeof(decltype(*matrix)));
    else{
      for (int i=0; i<n; i++){
        memset(matrix+i*lda,reset_val,m*sizeof(decltype(*matrix)));
      }
    }
    std::get<5>(t)(matrix,m,n,lda);
    invoke(ts...);// recursively invoke on next matrix
  }
};

template<typename func_type, typename... t1_types, typename... t2_types, size_t... index_list, typename... arg_types>
inline void selective_blas(std::string kernel_name, std::tuple<t1_types...>&& t1, std::tuple<t2_types...>&& t2,
                             IndexPack<index_list...>, func_type* func, arg_types... args){
/*
  if (internal::kernel_map.find(kernel_name) == internal::kernel_map.end()){
    float blas_confidence_tolerance_threshold = (std::getenv("SELECTIVEX_BLAS_CONFIDENCE_TOLERANCE") != NULL ? std::stof(std::getenv("SELECTIVEX_BLAS_CONFIDENCE_TOLERANCE")) : 0.05);
    size_t blas_min_num_executions = (std::getenv("SELECTIVEX_BLAS_MIN_NUM_EXECUTIONS") != NULL ? std::stof(std::getenv("SELECTIVEX_BLAS_MIN_NUM_EXECUTIONS")) : 3);
    float blas_min_execution_time = (std::getenv("SELECTIVEX_BLAS_MIN_EXECUTION_TIME") != NULL ? std::stof(std::getenv("SELECTIVEX_BLAS_MIN_EXECUTION_TIME")) : 1e-3);
    selectivex::register_kernel<sizeof...(index_list),selectivex::internal::NoOpModel>(kernel_name.c_str(),[](size_t*,size_t*,size_t*){return false;},true,true,blas_confidence_tolerance_threshold,blas_min_execution_time,blas_min_execution_time);
  }
*/
  if (selectivex::kernel_map.find(kernel_name) == selectivex::kernel_map.end()){
    func(args...);
    return;
  }
  assert(selectivex::kernel_map.find(kernel_name) != selectivex::kernel_map.end());
  // Extract features
  std::array<size_t,sizeof...(index_list)> feature_vector = {{(size_t)std::get<index_list>(t1)...}};
  // Initiate selective execution
  bool schedule_decision = should_observe(kernel_name.c_str(),&feature_vector[0]);
  if (schedule_decision){
    func(args...);
  }
  else{
    selectivex::predict(kernel_name.c_str(),&feature_vector[0]);
  }
  selectivex::observe(kernel_name.c_str(),&feature_vector[0]);
}

template<typename func_type, typename... t1_types, typename... t2_types, size_t... index_list1, typename... arg_types, size_t... index_list2, typename... TupleTypes>
inline int selective_lapack(std::string kernel_name, std::tuple<t1_types...>&& t1, std::tuple<t2_types...>&& t2,
                              IndexPack<index_list1...>, func_type* func, std::tuple<arg_types...>&& args, IndexPack<index_list2...>,
                              TupleTypes&&... reset_lambdas){
/*
  if (internal::kernel_map.find(kernel_name) == internal::kernel_map.end()){
    float lapack_confidence_tolerance_threshold = (std::getenv("SELECTIVEX_LAPACK_CONFIDENCE_TOLERANCE") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_CONFIDENCE_TOLERANCE")) : 0.05);
    size_t lapack_min_num_executions = (std::getenv("SELECTIVEX_LAPACK_MIN_NUM_EXECUTIONS") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_MIN_NUM_EXECUTIONS")) : 3);
    float lapack_min_execution_time = (std::getenv("SELECTIVEX_LAPACK_MIN_EXECUTION_TIME") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_MIN_EXECUTION_TIME")) : 1e-3);
    selectivex::register_kernel<sizeof...(index_list1),selectivex::internal::NoOpModel>(kernel_name.c_str(),[](size_t*,size_t*,size_t*){return false;},true,true,lapack_confidence_tolerance_threshold,lapack_min_num_executions,lapack_min_execution_time);
  }
*/
  if (selectivex::kernel_map.find(kernel_name) == selectivex::kernel_map.end()){
    assert(func(std::get<index_list2>(args)...)==0);
    return 0;
  }
  assert(selectivex::kernel_map.find(kernel_name) != selectivex::kernel_map.end());
  // Extract features
  std::array<size_t,sizeof...(index_list1)> feature_vector = {{(size_t)std::get<index_list1>(t1)...}};
  // Initiate selective execution
  bool schedule_decision = should_observe(kernel_name.c_str(),&feature_vector[0]);
  if (schedule_decision){
    reset_matrix::invoke(reset_lambdas...);
#ifdef SELECTIVEX__USE_MPI
    double runtime_start = MPI_Wtime();
#else
    auto runtime_start = std::chrono::high_resolution_clock::now();
#endif // SELECTIVEX__USE_MPI
    assert(func(std::get<index_list2>(args)...)==0);
#ifdef SELECTIVEX__USE_MPI
    double runtime = MPI_Wtime() - runtime_start;
#else
    auto runtime_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime_diff = runtime_end - runtime_start;
    double runtime = runtime_diff.count();
#endif // SELECTIVEX__USE_MPI
    selectivex::observe(kernel_name.c_str(),&feature_vector[0],runtime);
  }
  else{
    selectivex::predict(kernel_name.c_str(),&feature_vector[0]);
    selectivex::observe(kernel_name.c_str(),&feature_vector[0]);
  }
  return 0;// If not 0, an assert would be invoked

}

// Specialized overloads for complicated LAPACK routines.

template<typename T, typename func_type>
int selective_lapack_tpqrt(func_type* func, int matrix_layout, int m , int n , int l , int nb , T* a , int lda , T* b , int ldb , T* t , int ldt){
/*
  if (internal::kernel_map.find("dtpqrt") == internal::kernel_map.end()){
    float lapack_confidence_tolerance_threshold = (std::getenv("SELECTIVEX_LAPACK_CONFIDENCE_TOLERANCE") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_CONFIDENCE_TOLERANCE")) : 0.05);
    size_t lapack_min_num_executions = (std::getenv("SELECTIVEX_LAPACK_MIN_NUM_EXECUTIONS") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_MIN_NUM_EXECUTIONS")) : 3);
    float lapack_min_execution_time = (std::getenv("SELECTIVEX_LAPACK_MIN_EXECUTION_TIME") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_MIN_EXECUTION_TIME")) : 1e-3);
    selectivex::register_kernel<4,selectivex::internal::NoOpModel>("dtpqrt",[](size_t*,size_t*,size_t*){return false;},true,true,lapack_confidence_tolerance_threshold,lapack_min_num_executions,lapack_min_execution_time);
  }
*/
  assert(selectivex::kernel_map.find("dtpqrt") != selectivex::kernel_map.end());
  // Extract features
  std::array<size_t,4> feature_vector = {{(size_t)m,(size_t)n,(size_t)l,(size_t)nb}};
  // Initiate selective execution
  bool schedule_decision = should_observe("dtpqrt",&feature_vector[0]);
  if (schedule_decision){
/*
    for (int i=0; i<n; i++){
      memset(a+i*lda,1,(i+1)*sizeof(T));// Assumes column-major
      memset(a+i*lda+i+1,0,(n-i-1)*sizeof(T));// Assumes column-major
    }
    for (int i=0; i<n; i++){
      memset(b+i*ldb,1,(i+1)*sizeof(T));// Assumes column-major
      memset(b+i*ldb+i+1,0,(std::max(0,(m-l)-i-1))*sizeof(T));// Assumes column-major
      memset(b+i*ldb+(m-l),1,std::min(i+1,l)*sizeof(T));// Assumes column-major
      memset(b+i*ldb+(m-l)+i+1,0,std::max(0,(l-i-1))*sizeof(T));// Assumes column-major
    }
*/
#ifdef SELECTIVEX__USE_MPI
    double runtime_start = MPI_Wtime();
#else
    auto runtime_start = std::chrono::high_resolution_clock::now();
#endif // SELECTIVEX__USE_MPI
    assert(func(matrix_layout,m,n,l,nb,a,lda,b,ldb,t,ldt)==0);
#ifdef SELECTIVEX__USE_MPI
    double runtime = MPI_Wtime() - runtime_start;
#else
    auto runtime_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime_diff = runtime_end - runtime_start;
    double runtime = runtime_diff.count();
#endif // SELECTIVEX__USE_MPI
    selectivex::observe("dtpqrt",&feature_vector[0],runtime);
  } else{
    selectivex::predict("dtpqrt",&feature_vector[0]);
    selectivex::observe("dtpqrt",&feature_vector[0]);
  }
  return 0;// If not 0, an assert would be invoked
}

template<typename T, typename func_type>
int selective_lapack_tpmqrt(func_type* func, int matrix_layout, char side , char trans , int m , int n , int k , int l , int nb , const T* v ,
               int ldv , const T* t , int ldt , T* a , int lda , T* b , int ldb){
/*
  if (internal::kernel_map.find("dtpmqrt") == internal::kernel_map.end()){
    float lapack_confidence_tolerance_threshold = (std::getenv("SELECTIVEX_LAPACK_CONFIDENCE_TOLERANCE") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_CONFIDENCE_TOLERANCE")) : 0.05);
    size_t lapack_min_num_executions = (std::getenv("SELECTIVEX_LAPACK_MIN_NUM_EXECUTIONS") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_MIN_NUM_EXECUTIONS")) : 3);
    float lapack_min_execution_time = (std::getenv("SELECTIVEX_LAPACK_MIN_EXECUTION_TIME") != NULL ? std::stof(std::getenv("SELECTIVEX_LAPACK_MIN_EXECUTION_TIME")) : 1e-3);
    selectivex::register_kernel<5,selectivex::internal::NoOpModel>("dtpmqrt",[](size_t*,size_t*,size_t*){return false;},true,true,lapack_confidence_tolerance_threshold,lapack_min_num_executions,lapack_min_execution_time);
  }
*/
  assert(selectivex::kernel_map.find("dtpmqrt") != selectivex::kernel_map.end());
  // Extract features
  std::array<size_t,5> feature_vector = {{(size_t)m,(size_t)n,(size_t)k,(size_t)l,(size_t)nb}};
  // Initiate selective execution
  bool schedule_decision = should_observe("dtpmqrt",&feature_vector[0]);
  if (schedule_decision){
    T* v_temp = (T*)v;
    T* t_temp = (T*)t;
    if (side == 'L'){
      for (int i=0; i<k; i++){
        memset(v_temp+i*ldv,1,m*sizeof(T));// Assumes column-major
      }
    } else{
      for (int i=0; i<n; i++){
        memset(v_temp+i*ldv,1,n*sizeof(T));// Assumes column-major
      }
    }
/*
    for (int i=0; i<k; i++){
      memset(t_temp+i*ldt,1,nb*sizeof(T));// Assumes column-major
    }
    if (side=='L'){
      for (int i=0; i<n; i++){
        memset(a+i*lda,1,k*sizeof(T));// Assumes column-major
      }
    } else{
      for (int i=0; i<k; i++){
        memset(a+i*lda,1,m*sizeof(T));// Assumes column-major
      }
    }
    for (int i=0; i<n; i++){
      memset(b+i*ldb,1,m*sizeof(T));// Assumes column-major
    }
*/
#ifdef SELECTIVEX__USE_MPI
    double runtime_start = MPI_Wtime();
#else
    auto runtime_start = std::chrono::high_resolution_clock::now();
#endif // SELECTIVEX__USE_MPI
    assert(func(matrix_layout,side,trans,m,n,k,l,nb,v_temp,ldv,t_temp,ldt,a,lda,b,ldb)==0);
#ifdef SELECTIVEX__USE_MPI
    double runtime = MPI_Wtime() - runtime_start;
#else
    auto runtime_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime_diff = runtime_end - runtime_start;
    double runtime = runtime_diff.count();
#endif // SELECTIVEX__USE_MPI
    selectivex::observe("dtpmqrt",&feature_vector[0],runtime);
  } else{
    selectivex::predict("dtpmqrt",&feature_vector[0]);
    selectivex::observe("dtpmqrt",&feature_vector[0]);
  }
  return 0;// If not 0, an assert would be invoked
}

}

#endif // SELECTIVEX__INTERCEPT__FUNCGENERATORS_H_
