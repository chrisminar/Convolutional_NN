/***************************************************************************//**
 * \file train.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the training kernels
 */

#include "train.h"

namespace kernels
{

__global__
void calculate_dweight(double *dweight, double *input, double *ddot, int *pool_flag, int filter_size, int field_height, int field_width, int layer_depth, int layer_depth_out, int batch_size)
{
	int weight_index = threadIdx.x + blockDim.x * blockIdx.x,						//weight array index
		outer_layer_index = weight_index % (filter_size*filter_size*layer_depth),	//count from 0 to fs*fs*ld for each layer out
		outer_layer_number = weight_index / (filter_size*filter_size*layer_depth),	//outer layer this weight is in
		inner_layer_index = outer_layer_index % (filter_size*filter_size),			//count from 0 to fs*fs for each inner layer
		inner_layer_number = outer_layer_index / (filter_size*filter_size),			//inner layer this weight is in
		layer_y = inner_layer_index/filter_size,									//y position in layer
		layer_x = inner_layer_index%filter_size;									//x position in layer

	if (weight_index >= filter_size*filter_size*layer_depth*layer_depth_out)
		return;

	double sum=0;
	int input_index;
	int ddot_index;
	int flag=1;

	//step through images
	for (int m=0; m<batch_size; m++)
	{
		//step through field y
		for (int j=0; j<field_height; j++)
		{
			//step through field x
			for (int i=0; i<field_width; i++)
			{
				input_index =	m * field_width*field_height*layer_depth +
								inner_layer_num * field_width*field_height +
								j * field_width +
								i;
				ddot_index = 	m * field_width*field_height*layer_depth_out +
								outer_layer_num * field_width*field_height +
								j * field_width +
								i;

				//if were out of bounds, mult this weight*temp by zero
				if ( (layer_y == 0 and j==field_height-1) || (layer_y==filter_size-1 and j==0) || (layer_x == 0 and i==field_width-1) || (layer_x==filter_size-1 and i == 0) ) //todo might be able to cut out this if statement by modifiying the above for loops
					flag = 0;
				else
					flag = 1;
				sum += input[input_index] * ddot[ddot_index] * flag * pool_flag[i];
			}
		}
	}
	dweight[weight_index] = sum;
}

//can't handle things that are not 1x1xsomething
__global__
void calculate_fc_dweight(double *dweight, double *input, double *ddot, int filter_size, int field_height, int field_width, int layer_depth, int layer_depth_out, int batch_size)
{
	int weight_index = threadIdx.x + blockDim.x * blockIdx.x,						//weight array index
		outer_layer_index = weight_index % (filter_size*filter_size*layer_depth),	//count from 0 to fs*fs*ld for each layer out
		outer_layer_number = weight_index / (filter_size*filter_size*layer_depth),	//outer layer this weight is in
		inner_layer_index = outer_layer_index % (filter_size*filter_size),			//count from 0 to fs*fs for each inner layer
		inner_layer_number = outer_layer_index / (filter_size*filter_size),			//inner layer this weight is in
		layer_y = inner_layer_index/filter_size,									//y position in layer
		layer_x = inner_layer_index%filter_size;									//x position in layer

	if (weights >= filter_size*filter_size*layer_depth*layer_depth_out)
		return;

	double sum=0;
	int input_index;
	int ddot_index;

	//step through images
	for (int m=0; m<batch_size; m++)
	{
		ddot_index =	1*1*layer_depth_out*m + //1x1 should be be field width out and field height out
						1*1*outer_layer_number;
		input_index =	m * field_width*field_height*layer_depth +
						inner_layer_number * field_width*field_height +
						layer_y * field_width +
						layer_x;
		sum += input[input_index] * ddot[ddot_index];
	}
	dweight[neuron_index] = sum;
}

__global__
void propogate_ddot_fc()
{

}

//note: not setup for alternate filter sizes, zero padding or strides
__global__
void propogate_ddot_conv(double *ddot_current, double *ddot_past, double *weights, double *bias, int filter_size, int field_height, int field_width, int layer_depth, int layer_depth_out, int batch_size)
{
	//some useful numbers
	int num_pixels_per_layer = field_width*field_height,
		num_pixels_per_image = num_pixels_per_layer*layer_depth;

	//figure out where this kernel is
	int output_index = threadIdx.x + blockDim.x * blockIdx.x,		// ddot index
		image_number = output_index / (num_pixels_per_image),		// image this ddot is in
		image_index = filter_index % (num_pixels_per_image),		// count from 0 to num_pixels_per_image in this image
		layer_out_number = image_index / (num_pixels_per_layer),	// output layer this image is in
		layer_out_index = image_index % (num_pixels_per_layer),		// count from 0 to num_pixels_per_layer in this output layer
		field_x = layer_out_index % field_width,					// ddot x position in output layer
		field_y = layer_out_index / field_width;					// ddot y position in output layer


	//if were outside of the minibatch range return
	if (output_index >= num_pixels_per_image * batch_size)
		return;

	//setup dot product
	double sum = 0;
	int center_pixel_index = 0;
	int weight_index = 0;
	//the weight is going to be filter_size * filter_size * layer depth_in. There is one weight for each output layer
	//loop over output depth
	int filter_half = filter_size/2;
	//loop over input depth
	for (int k=0; k<layer_depth; k++)
	{
		center_pixel_index =	image_number*num_pixels_per_image + // past images
								k*num_pixels_per_layer +			// layers
								field_y*field_width +				// rows
								field_x;											// columns
		//loop over filter_x
		for (int i=-filter_size/2; i<filter_half; i++)
		{
			//loop over filter_y
			for (int j=-filter_size/2; j<filter_half; j++)
			{
				//check if we are at a boundary to account for zero padding
				//      left                           right                           bottom                     top
				if ( (filter_x + i < 0) || (filter_x + i >= field_width-1) || (filter_y + j < 0) || (filter_y + j >= field_height-1) )
				{}
				else
				{

					weight_index = 			layer_out_number*filter_size*filter_size*layer_depth +			// layer outs
											k*filter_size*filter_size +										// layer in
											filter_size*filter_size - 1 -									// maximum filter layer
											filter_size*(j+filter_half) -									// filter rows
											(i+filter_half);												// filter columns note: each filter is rotated by 180, which is why we go the the max of the filter layer then subtract off
					sum += ddot_current[center_pixel_index + j*field_width + i] * weights[weight_index];
				}//endif
			}//endj
		}//endi
	}//endk
	ddot_past[output_index] = sum + bias[m]; //todo deal with bias
}

}
