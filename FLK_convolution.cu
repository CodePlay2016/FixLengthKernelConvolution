#include "./FLK_convolution-inl.h"
#include <vector>

namespace mxnet {
namespace op {

  template<>
  Operator* CreateOp<gpu>(FixLengthKernelConvolutionParam param, int dtype,
    std::vector<TShape> *in_shape,
    std::vector<TShape> *out_shape,
    Context ctx) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new FixLengthKernelConvolutionOp<gpu, DType>(param);
    })
      return op;
  }

}  // namespace op
}  // namespace mxnet
