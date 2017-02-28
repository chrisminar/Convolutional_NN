/***************************************************************************//**
 * \file train.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementation of train epoch methods
 */

#include <network.h> //link to class namespace
#include <kernels/train.h> //link kernels

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
	thrust::plus<float> binary_op;
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

void network::ddot_conv_layer(int i)
{
	// convert temp into ddot
	//apply sigmoid to ddot
	kernels::d_sig_from_sig<double> dsig_op;
	//ddot = (1-temp)*temp
	thrust::transform(layers[i].temp.begin(), layers[i].temp.end(),					//iterate over temp
						layers[i].ddot.begin(),										//copy into ddot
						dsig_op);													//temp is the sigmoid of the convoluted input, take the sigmoid and make it into a derivative
}


void network::dw_conv(int i)
{
	const int blocksize = 256;
	dim3 grid( int((layers[i].filter_size*layers[i].filter_size*layers[i]*layer_depth*layers[i].layer_depth_out)/blocksize)+1, 1);
	dim3 block(blocksize,1);
	kernels::calculate_dweight<<<grid,block>>>(layers[i].dweight_r, layers[i].temp_r, layers[i].dtemp_r, layers[i].filter_size, layers[i].field_height, layers[i].field_width,
												layers[i].layer_depth, layers[i].layer_depth_out, batch_size); //todo I think this should be operating on the layer input, not the temp
}

/*
 * on a fully connected layer the size of dw is different);
 */
void network::dw_fc(int i)
{
	const int blocksize = 256;
	dim3 grid( int(layers[i].weights.size()/blocksize)+1, 1);
	dim3 block(blocksize,1);
	kernels::calculate_fc_dweight<<<grid,block>>>(layers[i].dweight, layers[i].input, layers[i].dinput, layers[i].filter_size, layers[i].field_height, layers[i].field_width,
													layers[i].layer_depth, layers[i].layer_depth_out, batch_size);
}
