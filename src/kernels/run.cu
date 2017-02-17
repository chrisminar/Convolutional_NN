/***************************************************************************//**
 * \file run.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the class layer
 */

#include "run.h"

namespace kernels
{
__global__
void convolute(int *input, int *output, double *weights, int field_width, int field_height,
		int stride_x, int stride_y, int zero_pad_x, int zero_pad_y, int filter_size, int batch_size, int layer_depth, int layer_depth_out)
{
	//todo currently assumes the same in x and y
	//todo currently can't handle zero_pad
	zero_pad_x = 0;
	//if were outside of the minibatch range return
	int number_of_filters_per_input = ((field_width - filter_size + 2*zero_pad_x)/stride_x + 1)^2;
	if (threadIdx.x + blockDim.x + blockIdx.x >= number_of_filters_per_input * layer_depth * batch_size)
		return;
	//figure out where this kernel is
	int total_filter_number = threadIdx.x + blockDim.x + blockIdx.x,
		image_number = total_filter_number/(number_of_filters_per_input),
		filter_number = total_filter_number%(number_of_filters_per_input),
		filter_x = filter_number % field_width,
		filter_y = filter_number / field_width;

	//setup dot product
	//this if statement causes crazy divergence
	double sum = 0;
	int center_pixel = 0;
	//the weight is going to be filter_size x filter_size x layer depth_in. There is one weight for each output layer
	//loop over depth
	for (int k=0; k<layer_depth; k++)
	{
		//loop over filter_x
		for (int i=-filter_width/2; i<filter_width/2; i++)
		{
			//loop over filter_y
			for (int j=-filter_width/2; j<filter_height/2; j++)
			{
				//check if we are at a boundary
				if (filter_x + i < 0)
				{
					//we're off the left side of the image, do nothing
				}
				else if (filter_x + i >= field_width)
				{
					//we're off the right side of the image, this doesn't really account for stride
				}
				else if (filter_y + j < 0)
				{
					//off the bottom
				}
				else if (filter_y + j > field_height)
				{
					//off the top
				}
				else
				{
					//otherwise we have no zero-padding required, dot product away
					center_pixel = image_number*field_width*field_height*k +  //you are here
					sum += input[] * weight[]
				}

			}
		}
	}


}

__global__
void dosomething(int stuff)
{
	if (threadIdx.x + blockDim.x + blockIdx.x >= nx*ny)
		return;
	int		ip = threadIdx.x + blockDim.x * blockIdx.x,
			I = ip% nx,
			J = ip / nx,
			iu = (nx-1)*J+I;
}
}
