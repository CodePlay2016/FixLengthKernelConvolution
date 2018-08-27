#include "./FLK_convolution_v4-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(FixLengthKernelConvolutionV4Param);

template<>
Operator* CreateOp<cpu>(FixLengthKernelConvolutionV4Param param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new FixLengthKernelConvolutionV4Op<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *FixLengthKernelConvolutionV4Prop::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}
/*
mxnet.ndarray.contrib.FixLengthKernelConvolutionV4(data=None, kernel_masks=None, weight=None, bias=None,
               kernel=_Null, kernel_max=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null,
			   workspace=_Null, no_bias=_Null, layout=_Null, out=None,
			   name=None, **kwargs)
*/

MXNET_REGISTER_OP_PROPERTY(_contrib_FixLengthKernelConvolutionV4, FixLengthKernelConvolutionV4Prop)
.describe(R"code(Compute 2-D fix length kernel ConvolutionV4 on 4-D input.

For 2-D deformable ConvolutionV4, the shapes are

- **data**: *(batch_size, channel, height, width)*
- **kernel_masks**: *(num_filter, channel, kernel_max)*
- **weight**: *(num_filter, channel, kernel_max)*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_height, out_width)*.

Define::

  f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1

then we have::

  out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
  out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])

If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

The default data ``layout`` is *NCHW*, namely *(batch_size, channle, height,
width)*.

If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
evenly into *g* parts along the channel axis, and also evenly split ``weight``
along the first dimension. Next compute the ConvolutionV4 on the *i*-th part of
the data with the *i*-th weight part. The output is obtained by concating all
the *g* results.

If ``num_deformable_group`` is larger than 1, denoted by *dg*, then split the
input ``offset`` evenly into *dg* parts along the channel axis, and also evenly
split ``out`` evenly into *dg* parts along the channel axis. Next compute the
deformable ConvolutionV4, apply the *i*-th part of the offset part on the *i*-th
out.


Both ``weight`` and ``bias`` are learnable parameters.


)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the FixLengthKernelConvolutionV4Op.")
.add_argument("kernel_masks", "NDArray-or-Symbol", "the kernel mask matrix.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(FixLengthKernelConvolutionV4Param::__FIELDS__());

}  // namespace op
}  // namespace mxnet
