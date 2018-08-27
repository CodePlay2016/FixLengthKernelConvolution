#include "./FLK_convolution_v2-inl.h"
#include <vector>

namespace mxnet {
namespace op {

  template<>
  Operator* CreateOp<gpu>(FixLengthKernelConvolutionV2Param param, int dtype,
    std::vector<TShape> *in_shape,
    std::vector<TShape> *out_shape,
    Context ctx) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new FixLengthKernelConvolutionV2Op<gpu, DType>(param);
    })
      return op;
  }

}  // namespace op
}  // namespace mxnet
