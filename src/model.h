#ifndef SELECTIVEX__MODEL_H_
#define SELECTIVEX__MODEL_H_

#include <vector>
#include <array>
#include <fstream>

namespace selectivex{
namespace internal{

template<typename ModelType>
class Model{
public:

  template<typename... ModelArgTypes>
  Model(ModelArgTypes&&... model_args);

  ~Model();

  float train();

  template<size_t nfeatures>
  float predict(std::array<size_t,nfeatures>& feature);

  void write_to_file();

  ModelType* _model;
  //std::ofstream write_file;
};

}
}

#endif /*SELECTIVEX__MODEL_H_*/
