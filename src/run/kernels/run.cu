/***************************************************************************//**
 * \file run.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the class layer
 */

#include "run.h"
#include <stdio.h> //used for printf
namespace kernels
{
__global__
void convolute(double *input, double *temp, double *weights, double *bias,
				int field_width, int field_height,
				int stride_x, int stride_y, int zero_pad_x, int zero_pad_y,
				int filter_size, int batch_size, int layer_depth, int layer_depth_out)
{
	//todo currently some parts assume the x-y symmetry
	//todo stride isn't used
	//todo zero pad is automatically set such that the output is the same size as the input
	//todo not sure if this will work with a filter size other than 3

	//some useful numbers
	int num_pixels_per_layer = field_width*field_height,
		num_pixels_per_image = num_pixels_per_layer*layer_depth_out;

	//figure out where this kernel is
	int temp_index = threadIdx.x + blockDim.x * blockIdx.x,
		image_number = temp_index / (num_pixels_per_image),
		image_index = temp_index % (num_pixels_per_image),
		layer_out_number = image_index / (num_pixels_per_layer),
		layer_out_index = image_index % (num_pixels_per_layer),
		field_x = layer_out_index % field_width,
		field_y = layer_out_index / field_width;

	//if were outside of the minibatch range return
	if (temp_index >= num_pixels_per_image * batch_size)
		return;

	//setup dot product
	//this if statement causes crazy divergence
	double sum = 0;
	int	center_pixel_index = 0,
		weight_index = 0,
		filter_half = filter_size/2;
	//loop over input depth
	for (int k=0; k<layer_depth; k++)
	{
		//loop over filter_x
		for (int i=-filter_size/2; i<=filter_half; i++)
		{
			//loop over filter_y
			for (int j=-filter_size/2; j<=filter_half; j++)
			{
				//check if we are at a boundary
				if (field_x + i < 0)
				{}//we're off the left side of the image, do nothing
				else if (field_x + i >= field_width-1)
				{}//we're off the right side of the image
				else if (field_y + j < 0)
				{}//off the bottom
				else if (field_y + j >= field_height-1)
				{}//off the top
				else
				{
					//otherwise we have no zero-padding required, dot product away
					center_pixel_index = image_number * field_width*field_height*layer_depth + //past images
											k * field_width*field_height +						//layers
											field_y * field_width +								//rows
											field_x;											//cols
					weight_index = layer_out_number * filter_size*filter_size*layer_depth +
									k*filter_size*filter_size +
									filter_size*(j+filter_half) +
									(i+filter_half);
					sum += input[center_pixel_index + j*field_width + i] * weights[weight_index];
				}//endif
			}//endj
		}//endi
	}//endk
	temp[temp_index] = sum + bias[image_number];
}

__global__
void convolute_FC(double *input, double *temp, double *weights, double *bias, int field_width, int field_height,
				int batch_size, int layer_depth, int layer_depth_out)
{
	//todo currently some parts assume the x-y symmetry

	//useful numbers
	int num_pixel_per_layer = field_width*field_height,
		num_pixel_per_image = num_pixel_per_layer*layer_depth;

	int neuron_index = threadIdx.x + blockDim.x * blockIdx.x,
		image_index = neuron_index / layer_depth_out,
		neuron = neuron_index%layer_depth_out;

	//if were outside of the minibatch range return
	if (neuron_index >= layer_depth_out * batch_size)
		return;

	//setup dot product
	//this if statement causes crazy divergence
	double sum = 0;
	int weight_index = 0;
	int pixel_index = 0;
	//loop over input depth
	for (int k=0; k<layer_depth; k++)
	{
		//loop over field_x
		for (int i=0; i<field_width; i++)
		{
			//loop over filter_y
			for (int j=0; j<field_height; j++)
			{
				weight_index = num_pixel_per_image*neuron +
								num_pixel_per_layer*k +
								field_width*j +
								i;
				pixel_index = num_pixel_per_image*image_index +
								num_pixel_per_layer*k +
								field_width * j +
								i;
				sum += input[pixel_index] * weights[weight_index];
			}//endj
		}//endi
	}//endk
	temp[neuron_index] = sum + bias[neuron];
}

__global__
void sigmoid_activation(double *temp, int field_width, int field_height, int layer_depth_out, int batch_size)
{
	int number_of_pixels_per_image = field_width*field_height*layer_depth_out;
	if (threadIdx.x + blockDim.x * blockIdx.x >= number_of_pixels_per_image * batch_size)
		return;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	temp[i] =  1 / (1 + exp(-temp[i]));
}

__global__
void pool_input(double *temp, double *output, int *pool_flag, int field_width, int field_height, int layer_depth_out, int batch_size)
{
	int number_of_pixels_per_image = field_width*field_height*layer_depth_out;
	if (threadIdx.x + blockDim.x * blockIdx.x >= number_of_pixels_per_image * batch_size)
		return;
	int		pool_index = threadIdx.x + blockDim.x * blockIdx.x,						//index for the pool output
			image_number = pool_index/number_of_pixels_per_image,					//what image are we at
			layer_number = (pool_index%number_of_pixels_per_image)/(field_width*field_height),	//what layer are we at
			layer_index = (pool_index%number_of_pixels_per_image)%(field_width*field_height),		//layer index from 0 to xx within each layer
			pool_x = layer_index % (field_width),									//the column we are on
			pool_y = layer_index / (field_width),									//the row we are on
			temp_index = image_number*field_width*field_height*layer_depth_out*4 +	//pixels from previous images...
							layer_number*field_width*field_height*4 +					//from previous layers
							pool_y*field_width*4 +									//rows
							pool_x*2;												//columns
	//perform pooling opertaion
	double max = temp[temp_index];
	int index = temp_index;
	if (temp[temp_index+1] > max)
	{
		max = temp[temp_index+1];
		index = temp_index+1;
	}
	if (temp[temp_index + field_width*2] > max)
	{
		max = temp[temp_index + field_width*2];
		index = temp_index + field_width*2;
	}
	if (temp[temp_index + field_width*2 + 1] > max)
	{
		max = temp[temp_index + field_width*2 + 1];
		index = temp_index + field_width*2 + 1;
	}
	output[pool_index] = max;
	pool_flag[index] = 1;
}
}
