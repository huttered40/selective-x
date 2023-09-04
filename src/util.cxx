#include <fstream>

#include "util.h"

namespace selectivex{

bool save_kernel_execution_decision;
double reconstructed_kernel_execution_time = 0;
double accelerated_kernel_execution_time = 0;
double reconstructed_execution_time = 0;
double accelerated_execution_time = 0;
int request_id = 100;

#ifdef SELECTIVEX__USE_MPI
double save_accelerated_kernel_execution_clock;
double save_reconstructed_kernel_execution_clock;
double save_accelerated_execution_clock;
double save_reconstructed_execution_clock;
#else
std::chrono::time_point<std::chrono::high_resolution_clock> save_accelerated_kernel_execution_clock;
std::chrono::time_point<std::chrono::high_resolution_clock> save_reconstructed_kernel_execution_clock;
std::chrono::time_point<std::chrono::high_resolution_clock> save_accelerated_execution_clock;
std::chrono::time_point<std::chrono::high_resolution_clock> save_reconstructed_execution_clock;
#endif // SELECTIVEX__USE_MPI

// List of automatically-intercepted BLAS/LAPACK/MPI kernels
std::array<std::string,6> list_of_blas_kernels = {{"dgemm","dtrmm","dtrsm","dsyrk","dsyr2k","dsymm"}};
std::array<std::string,9> list_of_lapack_kernels = {{"dgetrf","dpotrf","dtrtri","dgeqrf","dorgqr","dormqr","dgetri","dtpqrt","dtpmqrt"}};
std::array<std::string,18> list_of_mpi_kernels = {{"barrier","bcast","reduce","allreduce","gather","allgather","scatter","reduce_scatter","alltoall","gatherv","allgatherv","scatterv","alltoallv","send","recv","sendrecv","isend","irecv"}};

namespace internal{

#ifdef SELECTIVEX__USE_MPI
std::array<double,4> local_transfer_buffer;
std::array<double,4> remote_transfer_buffer;
double* nonblocking_buffer;
int* nonblocking_request;
std::map<int,std::pair<double*,int*>> save_nonblocking_info;
#endif // SELECTIVEX__USE_MPI

std::string data_folder_path = "";
bool within_window = false;
bool within_kernel = false;
bool automatic_start = false;
bool initialize_kernels = true;
size_t p2p_protocol_switch = INT_MAX;
size_t invocation_switch = 0;
float random_execution_percentage = .1;


kernel_feature_model::kernel_feature_model(size_t _num_samples, float _M1, float _M2, bool _is_steady, bool _is_active){
  this->is_steady=_is_steady;
  this->is_active=true;// Set to true regardless of what _is_active is
  this->num_samples=_num_samples;
  this->M1=_M1;
  this->M2=_M2;
  this->num_invocations=0;
}

kernel_feature_model::kernel_feature_model(const kernel_feature_model& copy){
  this->is_steady=copy.is_steady;
  this->is_active=copy.is_active;
  this->num_samples=copy.num_samples;
  this->num_invocations=copy.num_invocations;
  this->M1=copy.M1;
  this->M2=copy.M2;
}

float kernel_feature_model::get_estimate(){
  return this->M1;
}

float kernel_feature_model::get_variance(){
  if (this->num_samples <= 1) return 1000000.;
  return this->M2 / (this->num_samples-1);
}

float kernel_feature_model::get_standard_deviation(){
  return pow(this->get_variance(),1./2.);
}


void kernel_feature_model::update_model(double runtime){
  //if (update_analysis == 0) return;// no updating of analysis -- useful when leveraging data post-autotuning phase
  if (this->is_active == true){
    //this->num_schedules++;
    this->num_samples++;
    this->num_invocations++;
    size_t n1 = this->num_samples-1;
    size_t n = this->num_samples;
    float x = runtime;
    float delta = x - this->M1;
    float delta_n = delta / n;
    float delta_n2 = delta_n*delta_n;
    float term1 = delta*delta_n*n1;
    this->M1 += delta_n;
    this->M2 += term1;
    //std::cout << this->num_invocations << "," << this->num_samples << "," << this->M1 << "," << this->M2 << std::endl;
  } else{
  }
}

void kernel_feature_model::update(size_t remote_num_samples, float remote_M1, float remote_M2){
  size_t n1 = remote_num_samples;
  size_t n2 = this->num_samples;
  float delta = this->M1 - remote_M1;
  this->M1 = (n1*remote_M1 + n2*this->M1)/(n1+n2);
  this->M2 = remote_M2 + this->M2 + delta*delta*n1*n2/(n1+n2);
  this->num_samples += remote_num_samples;
}

void kernel_feature_model::write_header_to_file(std::ofstream& write_file, size_t nfeatures){
  write_file << "kernel_name";
  for (size_t i=0; i<nfeatures; i++){
    write_file << ",feature" << i+1;
  }
  write_file << ",num_samples,mean_execution_time,variance_execution_time,is_execution_time_steady,is_kernel_active\n";
}

void kernel_feature_model::write_to_file(std::ofstream& write_file){
  write_file << "," << this->num_samples << "," << this->M1 << "," << this->M2 << "," << this->is_steady << "," << this->is_active << std::endl;
}

void kernel_feature_model::read_from_file(std::istringstream& iss){
  std::string temp_feature;
  std::getline(iss,temp_feature,',');
  this->num_samples = stoul(temp_feature);
  std::getline(iss,temp_feature,',');
  this->M1 = stof(temp_feature);
  std::getline(iss,temp_feature,',');
  this->M2 = stof(temp_feature);
  std::getline(iss,temp_feature,',');
  this->is_steady = stoul(temp_feature);
  std::getline(iss,temp_feature,',');
  this->is_active = stoul(temp_feature);
  // Cognizant of the effects of reseting each kernel with existing samples to start
  //   off unsteady and active. Comment out the bottom two lines if not needed.
  this->is_steady = false;
  this->is_active = true;
  //std::cout << "What is this - " << this->num_samples << " " << this->M1 << " " << this->M2 << " " << this->is_steady << " " << this->is_active << std::endl;
}

/*
void ...{
  .. load to file ..
  .. probably want to always set to active, even if steady ..
}
*/

void kernel_feature_model::clear_model(){
//  this->hash_id = 0;
//  this->registered_channels.clear();
  this->is_steady = false;
  this->is_active = true;
  this->num_samples = 0;
  this->num_invocations = 0;
  this->M1=0;
  this->M2=0;
}

// *****************************************************************

size_t abstract_kernel_type::get_count_for_confidence_internal(kernel_feature_model& m){
  return (invocation_switch==1 ? m.num_invocations : 1);
/*
  int n = 1;
  switch (kernel_execution_count_mode){
    case -1:
      n = 1; break;
    case 0:
      n=1; break;
    case 1:
      n = p.num_schedules; break;// Per-process invocation count, matches that used to calculate sample variance.
    case 2:
      n = p.num_schedules; break;//TODO: In future, this should reference the CP array, or be passed it.
  }
  n = std::max(1,n);
  return n;
*/
}

float abstract_kernel_type::get_confidence_interval(kernel_feature_model& m){
  // returns confidence interval length with 95% confidence level
  size_t n = this->get_count_for_confidence_internal(m);
  return 1.96*m.get_standard_deviation() / pow(n*1.,1./2.);
}

void abstract_kernel_type::set_execution_time_steady(kernel_feature_model& m){
  if (m.is_steady == true) return;
  m.is_steady = ((this->get_confidence_interval(m) / m.get_estimate()) < this->_error_tolerance) &&
                 (m.num_samples >= this->_min_num_samples) &&
                 (m.get_estimate()*m.num_samples >= this->_min_execution_time);
  // if (m.is_steady == true) std::cout << "Steady! " << m.num_samples << " " << m.get_estimate() << " " << m.get_variance() << " " << this->get_confidence_interval(m) << " " << this->get_confidence_interval(m) / m.get_estimate() << std::endl;
}

void abstract_kernel_type::set_execution_time_active(kernel_feature_model& m){
  // Note: only valid for sequential execution. Parallel execution will not call this.
  m.is_active = !m.is_steady;
}

NoOpModel::NoOpModel(){}
float NoOpModel::Train(){return 0;}
float NoOpModel::Predict(){return -1;}


}

std::map<std::string,size_t> kernel_map;
std::vector<internal::abstract_kernel_type*> kernel_list;

}
