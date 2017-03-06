/***************************************************************************//**
 * \file train_test.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of training test kernels
 */

#include <thrust/functional.h>
#pragma once

namespace kernels
{
__global__
void calculate_dweight_test(double *dweight, double *input, double *ddot, int *pool_flag,
							int filter_size, int field_height, int field_width,
							int layer_depth, int layer_depth_out, int batch_size,
							int *weight_index_t, int *oli_t, int *oln_t, int *ili_t,
							int *iln_t, int *ly_t, int *lx_t,
							int *count, double *sum, int *input_index, int *ddot_index);
;__global__
void calculate_fc_dweight_test(double *dweight, double *input, double *ddot,
							int filter_size, int field_height, int field_width,
							int layer_depth, int layer_depth_out, int batch_size,
							int *wi_t, int *oli_t, int *oln_t, int *ili_t,
							int *iln_t, int *ly_t, int *lx_t,
							int *input_index_t, int *ddot_index_t);
__global__
void propogate_ddot_conv_test(double *ddot, double *ddot_upstream, double *weights,
								int field_height, int field_width, int layer_depth_out, int filter_size,
								int field_height_us, int field_width_us, int layer_depth_out_us, int batch_size,
								int *oi_t, int *in_t, int *ii_t, int *lun_t, int *lui_t, int *fx_t, int *fy_t,
								double *cpi_t, double *wi_t);

}
