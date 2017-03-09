/***************************************************************************//**
 * \file train.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementation of train epoch methods
 */

#include <network.h> //link to class namespace
#include "kernels/train.h" //link kernels
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

/*
 * propogate error up the network
 * previous layer delta = convolute(change in error from previous layer output, weights)
 * dE/dy^l-1 = convolute(dE/dx, weights)
 * ddot^l-1 = convolute(dE/d(convoluted input from this layer(not temp, but the thing between input and temp), weights)
 */
void network::propogate_error_handler(bool v, int i)
{
	if (v)
		std::cout << "Propogating ddot from layer" << i << "\n";

	//if we are on a fully connected layer, upstream shouldn't be convoluted for
	if (layers[i].lyr_conv == FULLY_CONNECTED)
	{
		upstream_ddot_fc(i);
	}
	//otherwise, call the convolution kernel
	else
	{
		upstream_ddot_conv(i);
	}
}

/*
 * calculate deltas at current layer
 * delta = error at current layer * sigmoid of convoluted (layer input)
 * dE/dx = dE/dy * sigma'(x)
 * ddot = ddot * dsig_from_sig(temp)
 */
void network::ddot_handler(bool v, int i)
{
	if (v)
		std::cout << "Calculating ddot from layer" << i << "\n";
	//if we are on the output layer, we need to calculate the initial ddot
	if (layers[i].lyr_typ == OUTPUT)
	{
		initial_ddot(i);
	}
	//once we have the initial ddot, apply the transformation
	update_ddot(i);
}

/*
 * calculate weight gradiants at current layer
 * weight delta(delta, layer input)
 * dE/dw(dE/dx, y^l-1)
 * dw(ddot, layer_input)
 */
void network::dw_handler(bool v, int i)
{
	if (v)
		std::cout << "Calculating dweight from layer" << i << "\n";
	//if were at a fully connected layer
	if (layers[i].lyr_conv == FULLY_CONNECTED)
	{
		dw_fc(i);
	}
	else
	{
		dw_conv(i);
	}
	//update weights
	//update dweight to reflect training rate and momentum: dweight = dweight*trainingrate
	thrust::transform(layers[i].dweight.begin(), layers[i].dweight.end(),
						layers[i].dweight.begin(),
						kernels::ax<double>(layers[i].learning_rate));

	//update weights with weight deltas
	thrust::transform(layers[i].weights.begin(), layers[i].weights.end(),
						layers[i].dweight.begin(),
						layers[i].weights.begin(),
						thrust::minus<double>());
}

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
	//std::cout<<layers[i].ddot.size()<<std::endl;
	//std::cout<<error<<std::endl;
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
