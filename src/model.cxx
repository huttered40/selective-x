// Note: as each will be a full specialization, this file need not be a header file.

#include "model.h"
#include "util.h"

#ifdef SELECTIVEX__USE_MLPACK
#include <mlpack.hpp>
#endif // SELECTIVEX__USE_MLPACK

namespace selectivex{
namespace internal{

// *****************************************************************
// Template specializations for supported ModelTypes

template<>
template<>
Model<NoOpModel>::Model(){
  this->_model = new NoOpModel();
/*
  std::string write_str = file_location + "_kt"
    +std::to_string(kernel_type)+"_st"+std::to_string(sample_type)
    +"_"+std::to_string(thread_count)+"threads.csv";
  this->write_file.open(write_str,std::fstream::app);//,std::ios_base::app);
*/
}

template<>
Model<NoOpModel>::~Model(){
  //this->write_file.close();
  delete this->_model;
}

template<>
float Model<NoOpModel>::train(){
/*
  for (auto& it : training_data){
    write_file << id << "," << m << "," << n << "," << k << "," << mean << "," << rel_std_dev << std::endl;
  }
*/
  return this->_model->Train();
}

//NOTE: In the implementation file, only fully-specialized class template definitions can exist
template<>
template<>
float Model<NoOpModel>::predict<1>(std::array<size_t,1>& feature){
  return this->_model->Predict();
}
template<>
template<>
float Model<NoOpModel>::predict<2>(std::array<size_t,2>& feature){
  return this->_model->Predict();
}
template<>
template<>
float Model<NoOpModel>::predict<3>(std::array<size_t,3>& feature){
  return this->_model->Predict();
}
template<>
template<>
float Model<NoOpModel>::predict<4>(std::array<size_t,4>& feature){
  return this->_model->Predict();
}
template<>
template<>
float Model<NoOpModel>::predict<5>(std::array<size_t,5>& feature){
  return this->_model->Predict();
}


template<>
void Model<NoOpModel>::write_to_file(){
}

}
}
