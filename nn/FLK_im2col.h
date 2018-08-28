#ifndef MXNET_OPERATOR_CONTRIB_NN_FLK_IM2COL_H_
#define MXNET_OPERATOR_CONTRIB_NN_FLK_IM2COL_H_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <cstring>
#include <vector>
#include "../../mxnet_op.h"

namespace mxnet {
namespace op {

/*!\brief
 * cpu function of deformable_im2col algorithm
 * \param s device stream
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape (#channels, output_im_height, output_im_width, ...)
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param data_col column buffer pointer
 */
template <typename DType>
inline void FLK_im2col_v1(mshadow::Stream<cpu>* s,
  const DType* data_im, const DType* kernel_mask, const TShape& kmshape,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride, const TShape& dilation,
  DType* data_col, bool flag) {
  if (2 == kernel_shape.ndim()) {
    LOG(FATAL) << "only implemented in GPU";
  } else {
    LOG(FATAL) << "not implemented";
  }
}

template <typename DType>
inline void FLK_im2col_v2(mshadow::Stream<cpu>* s,
  const DType* data_im, const DType* kernel_mask, const DType* weight, const TShape& kmshape,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride, const TShape& dilation,
  DType* data_col, bool flag) {
  if (2 == kernel_shape.ndim()) {
    LOG(FATAL) << "only implemented in GPU";
  } else {
    LOG(FATAL) << "not implemented";
  }
}

template <typename DType>
inline void FLK_im2col_v3(mshadow::Stream<cpu>* s,
  const DType* data_im, const DType* kernel_mask, const DType* weight, const TShape& kmshape,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride, const TShape& dilation,
  DType* data_col, bool flag) {
  if (2 == kernel_shape.ndim()) {
    LOG(FATAL) << "only implemented in GPU";
  } else {
    LOG(FATAL) << "not implemented";
  }
}

template <typename DType>
inline void FLK_im2col_v4(mshadow::Stream<cpu>* s,
  const DType* data_im, const DType* kernel_mask, const TShape& kmshape,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride, const TShape& dilation,
  DType* data_col, bool flag) {
  if (2 == kernel_shape.ndim()) {
    LOG(FATAL) << "only implemented in GPU";
  } else {
    LOG(FATAL) << "not implemented";
  }
}

/*!\brief
 * cpu function of deformable_col2im algorithm
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
inline void FLK_col2im(mshadow::Stream<cpu>* s,
  const DType* data_col, const DType* kernel_mask, const TShape& km_shape,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride,
  const TShape& dilation,
  DType* grad_im, OpReqType req) {
  LOG(FATAL) << "only implemented in GPU";
}


/*!\brief
 * cpu function of deformable_col2im_coord algorithm
 * \param s device stream
 * \param data_col start pointer of the column buffer to be filled
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param grad_offset pointer of the offset (C, H, W,...) in the offset batch
 */


}  // namespace op
}  // namespace mxnet
#ifdef __CUDACC__
#include "./FLK_im2col.cuh"
#endif
#endif  // MXNET_OPERATOR_CONTRIB_NN_FLK_IM2COL_H_
