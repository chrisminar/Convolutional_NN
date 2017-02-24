/***************************************************************************//**
 * \file train.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the training kernels
 */

#include "train.h"

namespace kernels
{
__global__
void delta_FC(double *delta, double *weights, double *bias, double *bias_delta, int field_width, int field_height,
				int layer_depth, int field_width_out, int field_height_out, int layer_depth_out, int batch_size)
{
	//todo currently some parts assume the x-y symmetry

	int neuron_index = threadIdx.x + blockDim.x * blockIdx.x,
		neuron = neuron_index%layer_depth_out;

	//if were outside of the minibatch range return
	if (neuron_index >= field_height*field_width_out*layer_depth_out * batch_size)
		return;

	double sum=0;
	int weight_index;

	//step through layer_depth
	for (int k=0; k<layer_depth; k++)
	{
		//step through layer x
		for (int i=0; i<field_width; i++)
		{
			//step though layer y
			for (int j=0; j<field_height; j++)
			{
				weight_index = 	neuron * layer_height*layer_width*layer_depth +
								k * field_width*field_height +
								j * field_width +
								i;
				sum += delta[neuron_index]*weight[weight_index];
			}
		}
	//update with bias
	}
	sum += bias[neuron_index]*delta[neuron_index];
	delta[neuron_index] = sum;
	//bias delta
	bias_delta[neuron_index] = bias[neuron_index]*delta[neuron_index];
}

__global__
void delta_pb(double *delta, double *weights, double *bias, double *bias_delta, int field_width, int field_height,
				int layer_depth, int field_width_out, int field_height_out, int layer_depth_out, int batch_size)
{
	//todo currently some parts assume the x-y symmetry

	int neuron_index = threadIdx.x + blockDim.x * blockIdx.x,
		neuron = neuron_index%layer_depth_out;

	//if were outside of the minibatch range return
	if (neuron_index >= field_height*field_width_out*layer_depth_out * batch_size)
		return;

	double sum=0;
	int weight_index;

	//step through layer_depth
	for (int k=0; k<layer_depth; k++)
	{
		//step through layer x
		for (int i=0; i<field_width; i++)
		{
			//step though layer y
			for (int j=0; j<field_height; j++)
			{
				weight_index = 	k * field_width*field_height +
								j * field_width +
								i;
				sum += delta[neuron_index]*weight[weight_index];
			}
		}
	//update with bias
	}
	sum += bias[neuron_index]*delta[neuron_index]; //todo set sum to zero somewhere
	delta[neuron_index] = sum;
	//bias delta
	bias_delta[neuron_index] = bias[neuron_index]*delta[neuron_index];
}

}
