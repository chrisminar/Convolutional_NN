/***************************************************************************//**
 * \file train.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the training kernels
 */

#include "train.h"

namespace kernels
{
__global__
void delta_FC(double *delta, double *weights, double *bias, int field_width, int field_height,
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
	if (neuron_index >= field_height*field_width_out*layer_depth_out * batch_size)
	return;

	double sum=0;
	int delta_index;
	int weight_index;
	delta_index = delta_index_start + neuron_index;
	//step through images
	for (int m=0; m<batch_size; m++)
	{
		//step through layer_depth
		for (int k=0; k<layer_depth; k++)
		{
		//step through layer x
		for (int i=0; i<field_width; i++)
		{
			//step though layer y
			for (int j=0; j<field_height; j++)
			{
				weight_index = something*image_start_number + //todo figure out this index
								something*m +
								field_width*field_height+k +
								field_width*j +
								i;
				sum += delta[delta_index]*weight[weight_index];
			}
		}
		//update bias
		sum += bias[k]*delta[delta_index];
	}
	}















	//setup dot product
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
}
