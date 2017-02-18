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
	int weight_index = 0;
	//the weight is going to be filter_size x filter_size x layer depth_in. There is one weight for each output layer
	//loop over output depth
	for (int m=0; m<layer_depth_out; m++)
	{
		//loop over input depth
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
						//	pixels from...                      past images							        past layers              past rows         past columns
						center_pixel_index = image_number*field_width*field_height*layer_depth +  field_width*field_height*k + field_width*filter_y + filter_x; //todo this only works if stride is one with approp zero-padding
						weight_index = filter_width*filter_height*layer_depth*m +  k*filter_width*filter_height + filter_width*j + i;
						sum += input[center_pixel_index + j*field_width + i] * weight[weight_index];
						//todo need to change inputs and outputs to be doubles
						//todo need to make a temporary array for the output of this step
					}//endif
				}//endj
			}//endi
		}//endk
		output[image_number*field_width*field_height*layer_depth_out + field_width*field_height*m + field_width * filter_y + filter_x] = sum;
	}//endm
}


__global__
void sigmoid_activation(int *output)
{
	int number_of_pixels_per_image = field_width*field_height*layer_depth_out;
	if (threadIdx.x + blockDim.x + blockIdx.x >= number_of_pixels_per_image * batch_size)
		return;
	int i = threadIdx.x + blockDim.x + blockIdx.x;
	output[i] =  1 / (1 + exp(-output[i]));
}

__global__
void pool_input(int *input, int *output)
{
	int number_of_pixels_per_image = field_width*field_height*layer_depth_out/4;
	if (threadIdx.x + blockDim.x + blockIdx.x >= number_of_pixels_per_image * batch_size)
		return;
	int		pool_index = threadIdx.x + blockDim.x + blockIdx.x,						//index for the pool output
			image_number = pool_index/number_of_pixels_per_image,					//what image are we at
			layer_number = (pool_index%number_of_pixels_per_image)/(field_width*field_height/4),	//what layer are we at
			layer_index = (pool_index%number_of_pixels_per_image)%(field_width*field_height/4),		//layer index from 0 to xx within each layer
			pool_x = layer_index % (field_width/4),									//the column we are on
			pool_y = layer_index / (field_width/4),									//the row we are on
			input_index = image_number*field_width*field_height*layer_depth_out +	//pixels from previous images...
							layer_number*field_width*field_height +					//from previous layers
							pool_y*field_width*2 +									//rows
							pool_x*2;												//columns
	//perform pooling opertaion
	double max = input[input_index];
	if (input[input_index+1] > max)
		max = input[input_index+1];
	if (input[input_index + field_width] > max)
		max = input[input_index + field_width];
	if (input[input_index + field_width + 1] > max)
		max = input[input_index + field_width + 1];
	output[pool_index] = max;


}
}
