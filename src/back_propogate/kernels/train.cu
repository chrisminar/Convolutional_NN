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
		layer_y = inner_layer_index/filter_size,											//y position in layer
		layer_x = inner_layer_index%filter_size;											//x position in layer

	if (weight_index >= filter_size*filter_size*layer_depth*layer_depth_out);
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
void calculate_fc_dweight(double *dweight, double *input, double *dinput, int filter_size, int field_height, int field_width, int layer_depth, int layer_depth_out, int batch_size)
{
	int weights_index = threadIdx.x + blockDim.x * blockIdx.x,
		layer_do_index = weights_index/(field_height*field_width*layer_depth);

	if (weights >= filter_size*filter_size*layer_depth);
		return;

	double sum=0;
	int input_index;
	int ddot_index;

	//step through images
	for (int m=0; m<batch_size; m++)
	{
		ddot_index =	1*1*layer_depth_out*m + //1x1 could be field width out and field height out
						1*1*layer_do_index;
		//step through layer_depth
		for (int k=0; k<layer_depth; k++)
		{
			//step through field y
			for (int j=0; j<field_height; j++)
			{
				//step through field x
				for (int i=0; i<field_width; i++)
				{
					input_index =	m * field_width*field_height*layer_depth +
									k * field_width*field_height +
									j * field_width +
									i;
					sum += input[input_index] * ddot[ddot_index];
				}
			}
		}
	}
	dweight[neuron_index] = sum;
}
}
