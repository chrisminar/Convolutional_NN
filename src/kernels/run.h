/***************************************************************************//**
 * \file run.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of run kernels
 */

#pragma once

namespace kernels
{
__global__
void convolute(double *input, double *temp, double *weights, double *bias, int field_width, int field_height,
		int stride_x, int stride_y, int zero_pad_x, int zero_pad_y, int filter_size, int batch_size, int layer_depth, int layer_depth_out);

__global__
void convolute_FC(double *input, double *temp, double *weights, double *bias, int field_width, int field_height,
				int batch_size, int layer_depth, int layer_depth_out);

__global__
void sigmoid_activation(double *temp, int field_width, int field_height, int layer_depth_out, int batch_size);

__global__
void pool_input(double *temp, double *output, int field_width, int field_height, int layer_depth_out, int batch_size);
}
