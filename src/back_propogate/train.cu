/***************************************************************************//**
 * \file train.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementation of train epoch methods
 */

#include <src/network.h> //link to class namespace
#include "kernels/train.h" //link kernels
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

/*
 * On the output layer, calculate the initial ddot by looking at the final results compared to the target
 * also calculate error
 */
void network::initial_ddot(int i)
{
	//note, can't handle FC with pooling
	//note I think this can only handle the last layer, possibly not all FC layers
	//ddot = network output - target
	thrust::transform(layers[i].layer_output.begin(), layers[i].layer_output.end(), //iterate over layer_output
						target.begin(), 											//subtract target
						layers[i].ddot.begin(), 									//place results in ddot
						thrust::minus<double>());									//subtract operator

	//Calculate L2 error: sum(ddot^2)
	kernels::square<double> unary_op;
	thrust::plus<double> binary_op;
	error = thrust::transform_reduce(layers[i].ddot.begin(), layers[i].ddot.end(),	//iterate over ddot
									unary_op,										//square ddot
									0.0,											//start sum at 0
									binary_op);										//reduction sum
}

/*
 * ddot = ddot*dsig(temp)
 */
void network::update_ddot(int i)
{
	kernels::sig_mult op;
	thrust::transform(layers[i].ddot.begin(), layers[i].ddot.end(),					//iterate over ddot
						layers[i].temp.begin(),										//start of temp
						layers[i].ddot.begin(),										//save in temp
						op);														//binary operator a*(1-b)*b
}

void network::dw_conv(int i)
{
	const int blocksize = 256;
	dim3 grid( int((layers[i].filter_size*layers[i].filter_size*layers[i].layer_depth*layers[i].layer_depth_out)/blocksize)+1, 1);
	dim3 block(blocksize,1);
	kernels::calculate_dweight<<<grid,block>>>(layers[i].dweight_r, layers[i].layer_input, layers[i].ddot_r, layers[i].pool_flag_r,
												layers[i].filter_size, layers[i].field_height, layers[i].field_width,
												layers[i].layer_depth, layers[i].layer_depth_out, batch_size);
}

/*
 * on a fully connected layer the size of dw is different);
 */
void network::dw_fc(int i)
{
	const int blocksize = 256;
	dim3 grid( int(layers[i].weights.size()/blocksize)+1, 1);
	dim3 block(blocksize,1);
	kernels::calculate_fc_dweight<<<grid,block>>>(layers[i].dweight_r, layers[i].layer_input, layers[i].ddot_r, layers[i].filter_size, layers[i].field_height, layers[i].field_width,
													layers[i].layer_depth, layers[i].layer_depth_out, batch_size);
}

void network::upstream_ddot_fc(int i)
{
	const int blocksize = 256;
	dim3 grid( int((layers[i-1].ddot.size())/blocksize)+1, 1);
	dim3 block(blocksize,1);
	kernels::propogate_ddot_fc<<<grid, block>>>(layers[i].ddot_r, layers[i-1].ddot_r, layers[i].weights_r, layers[i].bias_r,
															layers[i].field_height, layers[i].field_width, layers[i].layer_depth_out, layers[i].filter_size,
															layers[i-1].field_height, layers[i-1].field_width, layers[i-1].layer_depth_out,  batch_size);
}

void network::upstream_ddot_conv(int i)
{
	const int blocksize = 256;
	dim3 block(blocksize,1);
	dim3 grid( int((layers[i-1].ddot.size())/blocksize)+1, 1);
	if (i!=0)
	{
		kernels::propogate_ddot_conv<<<grid, block>>>(layers[i].ddot_r, layers[i-1].ddot_r, layers[i].weights_r, layers[i].bias_r,
														layers[i].field_height, layers[i].field_width, layers[i].layer_depth_out, layers[i].filter_size,
														layers[i-1].field_height, layers[i-1].field_width, layers[i-1].layer_depth_out,  batch_size);
	}
}
