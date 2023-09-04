#ifndef SELECTIVEX__ACCELERATE__RECORD__RECORD_H_
#define SELECTIVEX__ACCELERATE__RECORD__RECORD_H_

#include "../../util/util.h"

namespace selectivex{
namespace internal{
namespace accelerate{

class record{
public:
  static void write_file(int variantID, int print_mode, float overhead_time);
  static void print(int variantID, int print_mode, float overhead_time);

private:
  static void set_tuning_statistics();
  static void set_kernel_statistics();
};

}
}
}

#endif /*SELECTIVEX__ACCELERATE__RECORD__RECORD_H_*/
