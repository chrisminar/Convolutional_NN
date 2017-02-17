/***************************************************************************//**
 * \file run.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of run kernels
 */

#pragma once

namespace kernels
{
__global__
void convolute(int *input, int *output, double *weights, int filter_width, int filter_height,
		int stride_x, int stride_y, int filter_size, int batch_size, int layer_depth);

}
