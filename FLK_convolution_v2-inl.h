#ifndef MXNET_OPERATOR_CONTRIB_FLK_CONVOLUTION_V2_INL_H_
#define MXNET_OPERATOR_CONTRIB_FLK_CONVOLUTION_V2_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <time.h>
#include "../operator_common.h"
#include "../nn/im2col.h"
#include "./nn/FLK_im2col.h"
#include "../linalg.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

namespace conv {
  enum FixLengthKernelConvolutionV2OpInputs { kData, kKernelMasks, kWeight, kBias };
  enum FixLengthKernelConvolutionV2OpOutputs { kOut };
  enum FixLengthKernelConvolutionV2OpResource { kTempSpace };
}
struct FixLengthKernelConvolutionV2Param : public dmlc::Parameter<FixLengthKernelConvolutionV2Param> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint32_t kernel_max;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(FixLengthKernelConvolutionV2Param) {
    DMLC_DECLARE_FIELD(kernel).describe("ConvolutionV2 kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(kernel_max).describe("the selected kernel number");
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
      .describe("ConvolutionV2 stride: (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
      .describe("ConvolutionV2 dilate: (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
      .describe("Zero pad for ConvolutionV2: (h, w) or (d, h, w). Defaults to no padding.");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
      .describe("ConvolutionV2 filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
      .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
      .describe("Maximum temperal workspace allowed for ConvolutionV2 (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
      .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(layout)
      .add_enum("NCW", mshadow::kNCW)
      .add_enum("NCHW", mshadow::kNCHW)
      .add_enum("NCDHW", mshadow::kNCDHW)
      .set_default(dmlc::optional<int>())
      .describe("Set layout for input, output and weight. Empty for\n    "
        "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
  }
};

template<typename xpu, typename DType>
class FixLengthKernelConvolutionV2Op : public Operator {
 public:
  explicit FixLengthKernelConvolutionV2Op(FixLengthKernelConvolutionV2Param p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    CHECK(param_.layout.value() == mshadow::kNCW ||
      param_.layout.value() == mshadow::kNCHW ||
      param_.layout.value() == mshadow::kNCDHW)
      << "Only support NCW, NCHW and NCDHW layout";
  }

  virtual void Forward(const OpContext &ctx,
    const std::vector<TBlob> &in_data,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &out_data,
    const std::vector<TBlob> &aux_args) {

	clock_t sstart, start, end;
	sstart = clock();

    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[conv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 3 : 4;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    LayerSetUp(in_data[conv::kData].shape_,
               in_data[conv::kKernelMasks].shape_,
               out_data[conv::kOut].shape_);
    Stream<xpu>* s = ctx.get_stream<xpu>();
    // allocate workspace for both col_buffer and out_buffer, so *2 size
    Tensor<xpu, 1, DType> workspace = ctx.requested[conv::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s); 
    // calculate the shape of col_buffer
    TShape col_buffer_shape(num_spatial_axes_ + 1); // 3

	// add 'conv_out_channels_'
    col_buffer_shape[0] = conv_out_channels_ * conv_in_channels_ * param_.kernel_max;
    for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
      col_buffer_shape[i] = out_data[0].shape_[i + 1];
    }
    // create a column buffer using workspace and col_buffer_shape (c_out*c_in*kmax,H,W,)
    TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

    // initialize weight and col_buffer 3D tensors for using gemm
    index_t N = conv_out_spatial_dim_; // H*W
    index_t K = kernel_dim_; //cin/group*kmax

	Tensor<xpu, 3, DType> output_3d = out_data[conv::kOut].get_with_shape<xpu, 3, DType>(
      Shape3(num_, conv_out_channels_, N), s); // (B, Cout, H*W)
	
	//bool flag1 = true;
	//bool flag2 = true;
	bool flag = true;
    for (index_t n = 0; n < num_; ++n) {
      // transform image to col_buffer in order to use gemm
	  start = clock();
      FLK_im2col_v2(s, in_data[conv::kData].dptr<DType>() + n*input_dim_,
        in_data[conv::kKernelMasks].dptr<DType>(),
		in_data[conv::kWeight].dptr<DType>(),
		in_data[conv::kKernelMasks].shape_, in_data[conv::kData].shape_,
        col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
        col_buffer.dptr<DType>(), flag);
	  flag = false;
	  /*
	  if (flag1) {
		Tensor<cpu, 2, DType> out_buffer_2d_cpu = NewTensor<cpu, DType>(output_buffer_2d.shape_,DType(1));
		Copy(out_buffer_2d_cpu, output_buffer_2d, s);
		
		  for (index_t j=0; j<M*K; j++) {
		    for (index_t k=0; k<N; k++) {
		      DType tmp = out_buffer_2d_cpu[j][k];
			  LOG(INFO) << "print ["<< j << "],["<< k << "], it is " << tmp;
              //if (tmp >= -DBL_EPSILON && tmp <= DBL_EPSILON) {LOG(INFO) << "got zero at "<< i <<" "<< j <<" "<< k << ", it is " << tmp;
			  //break;}
			}
		  }
	  }
	  */
	  //flag1 = false;
	  //if (flag2) {
	  //  LOG(INFO) << "FLK_im2col time use " << (double)(clock()-start)/CLOCKS_PER_SEC;
	  //  flag2 = false;
	  //}
	  Tensor<xpu, 2, DType> output_2d = output_3d[n];
	  Tensor<xpu, 3, DType> output_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
	    Shape3(conv_out_channels_, K, N), s);
	  //Tensor<xpu, 3, DType> output_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
	    //Shape3(conv_out_channels_, K, N), s);
	  output_2d = reduce_with_axis<red::sum, false>(output_buffer_3d, 1);
    }
	end = clock();
	LOG(INFO) << "one convV2 time use " << (double)(end-sstart)/CLOCKS_PER_SEC;

    if (bias_term_) {
      Tensor<xpu, 1, DType> bias = in_data[conv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> output_3d = out_data[conv::kOut].get_with_shape<xpu, 3, DType>(
        Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      // has bias term, broadcast it to the same shape of output_3d in channel dim
      output_3d += mshadow::expr::broadcast<1>(bias, output_3d.shape_);
    }
  }

  virtual void Backward(const OpContext &ctx,
    const std::vector<TBlob>& out_grad,
    const std::vector<TBlob>& in_data,
    const std::vector<TBlob>& out_data,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& in_grad,
    const std::vector<TBlob>& aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    size_t expected = param_.no_bias == 0 ? 4 : 3;
    CHECK(in_data.size() == expected && in_grad.size() == expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[conv::kWeight].CheckContiguous(), true);
    LayerSetUp(in_grad[conv::kData].shape_,
               in_grad[conv::kKernelMasks].shape_,
               out_grad[conv::kOut].shape_);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // allocate workspace for col_buffer
    Tensor<xpu, 1, DType> workspace = ctx.requested[conv::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
    // calculate the shape of col_buffer
    TShape col_buffer_shape(num_spatial_axes_ + 1);
    col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
    for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
      col_buffer_shape[i] = out_grad[conv::kData].shape_[i + 1];
    }
    // create a column buffer using workspace and col_buffer_shape
    TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

    // initialize weight and col_buffer 3D tensors for using gemm
    // For computing dLoss/d(in_data[kData])
    index_t M = kernel_dim_; // Cin / group * kmax
    index_t N = conv_out_spatial_dim_; // H*W
    index_t K = conv_out_channels_ / group_;
    Tensor<xpu, 3, DType> weight_3d = in_data[conv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, M), s);
    Tensor<xpu, 4, DType> out_grad_4d = out_grad[conv::kOut].get_with_shape<xpu, 4, DType>(
      Shape4(num_, group_, K, N), s);
    Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
      Shape3(group_, M, N), s);
    // For computing dLoss/dWeight
    Tensor<xpu, 3, DType> dweight_3d = in_grad[conv::kWeight].get_with_shape<xpu, 3, DType>(
      Shape3(group_, K, M), s);

    Tensor<xpu, 1, DType> data_grad = in_grad[conv::kData].FlatTo1D<xpu, DType>(s);
    data_grad = 0;
	bool flag = true;


    for (index_t n = 0; n < num_; ++n) {
      Tensor<xpu, 3, DType> out_grad_3d = out_grad_4d[n];
      for (index_t g = 0; g < group_; ++g) {
        // Legacy approach shown here for comparison:
        //   col_buffer_3d[g] = dot(weight_3d[g].T(), out_grad_3d[g]);
        linalg_gemm(weight_3d[g], out_grad_3d[g], col_buffer_3d[g], true, false, s);
      }

      // gradient w.r.t. input data
      FLK_col2im(s, col_buffer.dptr<DType>(),
        in_data[conv::kKernelMasks].dptr<DType>() + n*kernel_masks_dim_,
        in_data[conv::kKernelMasks].shape_, in_grad[conv::kData].shape_, col_buffer.shape_,
        param_.kernel, param_.pad, param_.stride, param_.dilate,
        in_grad[conv::kData].dptr<DType>() + n*input_dim_,
        req[conv::kData]);

      // gradient w.r.t. weight, dWeight should accumulate across the batch and group
      FLK_im2col_v2(s, in_data[conv::kData].dptr<DType>() + n*input_dim_,
              in_data[conv::kKernelMasks].dptr<DType>() + n*kernel_masks_dim_,
              in_data[conv::kWeight].dptr<DType>() + n*kernel_masks_dim_,
      		in_data[conv::kKernelMasks].shape_, in_data[conv::kData].shape_,
              col_buffer.shape_, param_.kernel, param_.pad, param_.stride, param_.dilate,
              col_buffer.dptr<DType>(), flag);

      for (index_t g = 0; g < group_; ++g) {
        auto request = (n == 0) ? req[conv::kWeight] : kAddTo;
        // Legacy approach shown here for comparison:
        //   Assign(dweight_3d[g], request, dot(out_grad_3d[g], col_buffer_3d[g].T()));
        linalg_gemm(out_grad_3d[g], col_buffer_3d[g], dweight_3d[g], false, true, s, request);
      }
    }

    // gradient w.r.t bias
    if (bias_term_) {
      Tensor<xpu, 1, DType> dbias = in_grad[conv::kBias].get<xpu, 1, DType>(s);
      Tensor<xpu, 3, DType> dout = out_grad[conv::kOut].get_with_shape<xpu, 3, DType>(
        Shape3(num_, conv_out_channels_, conv_out_spatial_dim_), s);
      ASSIGN_DISPATCH(dbias, req[conv::kBias], sumall_except_dim<1>(dout));
    }
  }

 private:
  void LayerSetUp(const TShape& ishape, const TShape& kernel_masks_shape, const TShape& oshape) {
    channel_axis_ = 1;  // hard code channel axis
    const index_t first_spatial_axis = channel_axis_ + 1; // 2
    const index_t num_axes = param_.kernel.ndim() + 2; // 2/1
    num_spatial_axes_ = num_axes - first_spatial_axis;
    is_1x1_ = true;
    for (index_t i = 0; i < param_.kernel.ndim(); ++i) {
      is_1x1_ &= param_.kernel[i] == 1 && param_.stride[i] == 1 && param_.pad[i] == 0;
      if (!is_1x1_) break;
    }

    // batch size
    num_ = ishape[0];
    // number of input channels
    channels_ = ishape[1];
    group_ = param_.num_group;
    conv_out_channels_ = param_.num_filter;
    conv_in_channels_ = channels_;
    bias_term_ = !param_.no_bias;
	kernel_max_ = param_.kernel_max;
    kernel_dim_ = conv_in_channels_ / group_ * kernel_max_; // kernel.Size():kh*kw
    weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    conv_out_spatial_dim_ = oshape.ProdShape(2, oshape.ndim()); // H*W
    col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
    // size of the column buffer used for storing im2col-ed pixels
    col_buffer_size_ = conv_out_channels_ * kernel_dim_ * group_ * conv_out_spatial_dim_;
    // input/output image size (#channels * height * width)
    input_dim_ = ishape.ProdShape(1, ishape.ndim());
    kernel_masks_dim_ = kernel_masks_shape.ProdShape(1, kernel_masks_shape.ndim());
    output_dim_ = oshape.ProdShape(1, oshape.ndim());
    num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
    num_kernels_col2im_ = input_dim_;
  }

 private:
  FixLengthKernelConvolutionV2Param param_;
  index_t channel_axis_;  // channel axis of the input
  index_t channels_;  // number of channels of input image
  index_t num_spatial_axes_;  // number of spatial axes
  index_t num_;  // batch size
  index_t group_;  // number of groups
  index_t conv_out_channels_;  // number of output channels (num_filter)
  index_t conv_out_spatial_dim_;  // number of pixels of output images per channel
  index_t conv_in_channels_;  // number of input channels
  index_t kernel_dim_;  // number of input channels per group * kernel size
  index_t kernel_max_;
  index_t weight_offset_;  // number of output channels per group * kernel_dim_
  index_t col_offset_;
  index_t output_offset_;
  index_t col_buffer_size_;
  index_t input_dim_;
  index_t kernel_masks_dim_;
  index_t output_dim_;
  index_t num_kernels_im2col_;
  index_t num_kernels_col2im_;
  bool bias_term_;  // has bias term?
  bool is_1x1_;
};  // class ConvolutionV2Op

template<typename xpu>
Operator* CreateOp(FixLengthKernelConvolutionV2Param param, int dtype,
  std::vector<TShape> *in_shape,
  std::vector<TShape> *out_shape,
  Context ctx);

#if DMLC_USE_CXX11
class FixLengthKernelConvolutionV2Prop : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return{ "data", "kernel_masks", "weight", "bias" };
    } else {
      return{ "data", "kernel_masks", "weight" };
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 2) {
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      LOG(FATAL) << "not implemented";
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
    std::vector<TShape> *out_shape,
    std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 4U) << "Input:[data, offset, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 3U) << "Input:[data, offset, weight]";
    }
    out_shape->resize(1, TShape());
    const TShape &dshp = (*in_shape)[conv::kData];
    const TShape &wshp = (*in_shape)[conv::kWeight];
    if (dshp.ndim() == 0) return false;
    if (param_.kernel.ndim() == 2) {
      // 2d conv
      CHECK_EQ(dshp.ndim(), 4U) \
        << "Input data should be 4D in batch-num_filter-y-x";
      CHECK_EQ(wshp.ndim(), 3U) \
        << "Input weight should be 3D in num_filter-Cin-kmax";
      Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
      Shape<3> wshape = Shape3(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
        param_.kernel_max);
      wshape[0] *= param_.num_group;
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
      if (!param_.no_bias) {
        SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
      }

      const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
      const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
      CHECK_EQ(dshape[1] % param_.num_group, 0U) \
        << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
        << "output num_filter must divide group size";
      CHECK_GT(param_.kernel.Size(), 0U) \
        << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0U) \
        << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0U) \
        << "incorrect dilate size: " << param_.dilate;
      Shape<4> oshape;
      oshape[0] = dshape[0];
      oshape[1] = param_.num_filter;
      oshape[2] = (dshape[2] + 2 * param_.pad[0] -
        (param_.dilate[0] * (ksize_y - 1) + 1)) / param_.stride[0] + 1;
      oshape[3] = (dshape[3] + 2 * param_.pad[1] -
        (param_.dilate[1] * (ksize_x - 1) + 1)) / param_.stride[1] + 1;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
      CHECK_EQ(oshape[1] % param_.num_group, 0U) \
        << "output num_filter must divide deformable group size";
      // Perform incomplete shape inference. Fill in the missing values in data shape.
      // 1) We can always fill in the batch_size.
      // 2) We can back-calculate the input height/width if the corresponding stride is 1.
      oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
      dshape[0] = oshape[0];
      if (param_.stride[0] == 1) {
        dshape[2] = oshape[2] + param_.dilate[0] * (ksize_y - 1) - 2 * param_.pad[0];
      }
      if (param_.stride[1] == 1) {
        dshape[3] = oshape[3] + param_.dilate[1] * (ksize_x - 1) - 2 * param_.pad[1];
      }
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kData,
        ConvertLayout(dshape, kNCHW, param_.layout.value()));
      // Check whether the kernel sizes are valid
      if (dshape[2] != 0) {
        CHECK_LE(ksize_y, dshape[2] + 2 * param_.pad[0]) << "kernel size exceed input";
      }
      if (dshape[3] != 0) {
        CHECK_LE(ksize_x, dshape[3] + 2 * param_.pad[1]) << "kernel size exceed input";
      }
      return true;
    } else {
      LOG(FATAL) << "not implemented";
      return false;
    }
  }

  bool InferType(std::vector<int> *in_type,
    std::vector<int> *out_type,
    std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new FixLengthKernelConvolutionV2Prop();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_FixLengthKernelConvolutionV2";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return{ out_grad[conv::kOut], in_data[conv::kData],
            in_data[conv::kKernelMasks], in_data[conv::kWeight] };
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return{ ResourceRequest::kTempSpace };
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    return{ ResourceRequest::kTempSpace };
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const override;

 private:
  FixLengthKernelConvolutionV2Param param_;
};  // class ConvolutionV2Prop
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_FLK_ConvolutionV2_INL_H_
