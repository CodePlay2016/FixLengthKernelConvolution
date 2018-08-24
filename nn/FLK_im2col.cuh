#ifndef MXNET_OPERATOR_CONTRIB_NN_FLK_IM2COL_CUH_
#define MXNET_OPERATOR_CONTRIB_NN_FLK_IM2COL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include "../../mxnet_op.h"
#include "../../../common/cuda_utils.h"

namespace mxnet {
namespace op {

/*!
 * \brief FLK_im2col gpu kernel.
 * DO NOT call this directly. Use wrapper function im2col() instead;
 */
template <typename DType>
__global__ void FLK_im2col_gpu_kernel(const int n, const DType* data_im, // (Cin,H,W)
  const DType* kernel_mask, const DType* weight, // kernel mask (Cout,Cin,k_max)
  const int height, const int width, const int kernel_max, // pruned kernel size
  const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int channel_in,  const int channel_out,
  const int height_col, const int width_col,
  DType* data_col, bool flag) { // (c_out*c_in*kmax,H,W,)
  CUDA_KERNEL_LOOP(index, n) { // n is total kernel number，here it is cout*cin*H*W
	// index index of output matrix
    const int w_col = index % width_col; // index element in a certain width
    const int h_col = (index / width_col) % height_col; // index a width in a certain Height
    const int c_im = ((index / width_col) / height_col) % channel_in; // index a Height in a certain Cin
    const int c_col_out = ((index / width_col) / height_col) / channel_in; // index a Cin in a certain Cout/ index Cout

    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    DType* data_col_ptr = data_col + ((c_col_out * channel_in + c_im) * kernel_max * height_col + h_col) * width_col + w_col;
    const DType* data_im_ptr = data_im + (c_im * height + h_offset) * width + w_offset;
    const DType* kernel_mask_ptr = kernel_mask + (c_col_out * channel_in + c_im) * kernel_max;
	const DType* weight_ptr = weight + (c_col_out * channel_in + c_im) * kernel_max;

    for (int k = 0; k < kernel_max; ++k) {
		int k_index = (int) kernel_mask_ptr[k];
		DType w_value = weight_ptr[k];
		int i = k_index / kernel_w; // index in the domain of kernel_h*kernel_w
		int j = k_index % kernel_w;
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] * w_value : static_cast<DType>(0);
		/*
		if (flag && !(index % 100)) {
			printf("**data_col_ptr with index of %d/%d is %d, value of it is %f, k_index is %d, kernel_mask_ptr is %d, data_im_ptr value is: %f\n"
                  "c_col_out: %d, c_im: %d, h_col: %d, w_col: %d, h_im: %d, w_im: %d, height:%d, width:%d, data_col at this pos is %f, weight at this pos is %f\n",
                  index, n, data_col_ptr-data_col, *data_col_ptr, 
				  k_index, (c_col_out * channel_in + c_im), data_im_ptr[i*dilation_h*width+j*dilation_w],
				  c_col_out, c_im, h_col, w_col, h_im, w_im, height,
				  width, data_col[((c_col_out*channel_in+c_im)*kernel_max*height_col+h_col)*width_col+w_col], w_value);
		}
		*/
        data_col_ptr += height_col * width_col;
	}
  }
}

/*!
 * \brief FLK_im2col gpu kernel. version2
 * DO NOT call this directly. Use wrapper function im2col() instead;
 */
template <typename DType>
__global__ void FLK_im2col_gpu_kernel2(const int n, const DType* data_im, // (Cin,H,W)
  const DType* kernel_mask, // kernel mask (Cout,Cin,k_max)
  const int height, const int width, const int kernel_max, // pruned kernel size
  const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int channel_in,  const int channel_out,
  const int height_col, const int width_col,
  DType* data_col, bool flag) { // (c_out*c_in*kmax,H,W,)
  CUDA_KERNEL_LOOP(index, n) { // n is total kernel number，here it is cout*cin*H*W
	// index index of output matrix
	DType* ptr = data_col + index*kernel_max;
    for (int k = 0; k < kernel_max; k++) {
		*ptr = DType(0.2);
	}
  }
}



/*!\brief
 * gpu function of fix length kernel im2col algorithm
 * \param s device stream
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param filter_mask pointer of mask (Cout, Cin, kmax,)
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape (Cout, Cin, output_im_height, output_im_width, ...)
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param data_col column buffer pointer
 */
template <typename DType>
inline void FLK_im2col(mshadow::Stream<gpu>* s,
  const DType* data_im, const DType* kernel_mask, const DType* weight, const TShape& km_shape,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride, const TShape& dilation,
  DType* data_col, bool flag) {
  // num_axes should be smaller than block size
  index_t num_spatial_axes = kernel_shape.ndim();
  CHECK_LT(num_spatial_axes, mshadow::cuda::kBaseThreadNum);
  index_t num_kernels = km_shape[0] * km_shape[1] * col_shape.ProdShape(1, col_shape.ndim());
  using namespace mxnet_op;
  switch (num_spatial_axes) {
  case 2:
	//if (flag)
	//	printf("col_shape: %d, %d, %d\nkm_shape: %d, %d, %d\n",
	//           (int)col_shape[0], (int)col_shape[1], (int)col_shape[2],
	//		   (int)km_shape[0], (int)km_shape[1], (int)km_shape[2]);
    FLK_im2col_gpu_kernel<DType> // NOLINT_NEXT_LINE(whitespace/operators)
        <<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
           0, mshadow::Stream<gpu>::GetStream(s)>>>(
        num_kernels, data_im, kernel_mask, weight, im_shape[2], im_shape[3], 
		km_shape[2], kernel_shape[0], kernel_shape[1], 
        pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        km_shape[1], km_shape[0], col_shape[1], col_shape[2], data_col, flag);
    MSHADOW_CUDA_POST_KERNEL_CHECK(FLK_im2col_gpu_kernel);
    break;
  default:
    LOG(FATAL) << "im2col_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
}


/*!
* \brief deformable_col2im gpu kernel.
* \brief DO NOT call this directly. Use wrapper function deformable_col2im() instead;
*/
template <typename DType>
__global__ void FLK_col2im_gpu_kernel(const int n, const DType* data_col, const DType* kernel_masks,
  const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int height_col, const int width_col,
  DType* grad_im, OpReqType req) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col) % kernel_w;
    const int i = (index / width_col / height_col / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / kernel_w / kernel_h;
    // compute the start and end of the output

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const DType* kernel_masks_ptr = kernel_masks + 2 * kernel_h * kernel_w * height_col * width_col;
    const int kernel_masks_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int kernel_masks_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const DType offset_h = kernel_masks_ptr[kernel_masks_h_ptr];
    const DType offset_w = kernel_masks_ptr[kernel_masks_w_ptr];
    const DType cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const DType cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const DType cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
          cur_w + dx >= 0 && cur_w + dx < width &&
          abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
          abs(cur_inv_w_data - (cur_w + dx)) < 1
          ) {
          int cur_bottom_grad_pos = (c * height + cur_h + dy) * width + cur_w + dx;
          // DType weight = get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          // atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
  }
	

/*!\brief
 * gpu function of FLK_col2im algorithm
 * \param s device stream
 * \param data_col start pointer of the column buffer to be filled
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param grad_im pointer of a image (C, H, W,...) in the image batch
 */
template <typename DType>
inline void FLK_col2im(mshadow::Stream<gpu>* s,
  const DType* data_col, const DType* kernel_masks, const TShape& kmshape,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride,
  const TShape& dilation,
  DType* grad_im, OpReqType req) {
  index_t num_spatial_axes = kernel_shape.ndim();
  index_t im_size = im_shape.ProdShape(1, im_shape.ndim());
  index_t num_kernels = col_shape.ProdShape(0, col_shape.ndim());
  // num_axes should be smaller than block size
  CHECK_LT(num_spatial_axes, mshadow::cuda::kBaseThreadNum);
  using namespace mxnet_op;
  switch (num_spatial_axes) {
  case 2:
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)
    FLK_col2im_gpu_kernel<DType><<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
                               0, mshadow::Stream<gpu>::GetStream(s)>>>(
        num_kernels, data_col, kernel_masks, im_shape[1], im_shape[2], im_shape[3],
        kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
        dilation[0], dilation[1], col_shape[1], col_shape[2], grad_im, req);
    MSHADOW_CUDA_POST_KERNEL_CHECK(FLK_col2im_gpu_kernel);
    break;
  default:
    LOG(FATAL) << "col2im_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
}
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_NN_FLK_IM2COL_CUH_
