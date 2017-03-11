/***************************************************************************//**
 * \file run_methods.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementation of train epoch methods
 */

#include <network.h> //link to class namespace
#include "kernels/run.h" //link kernels
#include <io.h>
#include <thrust/extrema.h>//keep
#include <thrust/functional.h>//keep
#include <thrust/transform.h>//keep

void network::run_recast(int i)
{
	//resize stuff, not sure why this needs to be resized here but it breaks otherwise
	if (i>0)
	{
		layers[i].layer_input = thrust::raw_pointer_cast(&layers[i-1].layer_output[0]);
	}
	layers[i].temp_r = thrust::raw_pointer_cast( &(layers[i].temp[0]) );
	layers[i].layer_output_r = thrust::raw_pointer_cast( &(layers[i].layer_output[0]) );
	layers[i].pool_flag_r = thrust::raw_pointer_cast( &(layers[i].pool_flag[0]) );
	layers[i].weights_r = thrust::raw_pointer_cast( &(layers[i].weights[0]) );
}

void network::run_convolute(int i)
{
	if (verbose)
		std::cout<<"\tConvoluting\n";

	const int blocksize = 256;
	dim3 block (blocksize,1);

	//convolute each layer
	if (layers[i].lyr_conv == CONVOLUTIONAL)
	{
		dim3 conv_g( int( (layers[i].field_width*layers[i].field_height*layers[i].layer_depth_out*batch_size - 0.5)/blocksize ) + 1, 1);
		kernels::convolute<<<conv_g,block>>>(layers[i].layer_input, layers[i].temp_r, layers[i].weights_r, layers[i].bias_r,
											layers[i].field_width, layers[i].field_height, layers[i].stride_x, layers[i].stride_y,
											layers[i].zero_pad_x, layers[i].zero_pad_y, layers[i].filter_size, batch_size,
											layers[i].layer_depth, layers[i].layer_depth_out);
	}
	else if (layers[i].lyr_conv == FULLY_CONNECTED)
	{
		dim3 conv_g( int((layers[i].layer_depth_out*batch_size)/blocksize) +1, 1);
		kernels::convolute_FC<<<conv_g,block>>>(layers[i].layer_input, layers[i].temp_r, layers[i].weights_r, layers[i].bias_r,
											layers[i].field_width, layers[i].field_height, batch_size,
											layers[i].layer_depth, layers[i].layer_depth_out);
	}
	if (verbose)
		std::cout<<"\tConvoluted\n";
	if (output)
	{
		std::cout<<"printing conv from layer " << i <<std::endl;
		io::print_temp(1,0, layers[i], "convolute");
	}
}

void network::run_activation(int i)
{
	if (verbose)
		std::cout<<"\tActivating\n";
	const int blocksize = 256;
	dim3 block (blocksize,1);
	dim3 actv_g( int( (layers[i].field_width*layers[i].field_height*layers[i].layer_depth_out*batch_size)/blocksize ) +1, 1);

	//activation function
	kernels::sigmoid_activation<<<actv_g, block>>>(layers[i].temp_r, layers[i].field_width, layers[i].field_height,
													layers[i].layer_depth_out, layers[i].batch_size);

	if (verbose)
		std::cout<<"\tActivated\n";
	if (output)
	{
		std::cout<<"printing sigm from layer " << i <<std::endl;
		io::print_temp(1,0, layers[i], "sigmoid_activation");
	}
}

void network::run_pool(int i)
{
	if (verbose)
		std::cout<<"\tPooling\n";

	const int blocksize = 256;
	dim3 block (blocksize,1);
	dim3 pool_g ( int( (layers[i].field_width_out*layers[i].field_height_out*layers[i].layer_depth_out*batch_size)/blocksize ) +1, 1);

	//pool layer
	if (layers[i].pool)
	{
		kernels::pool_input<<<pool_g,block>>>(layers[i].temp_r, layers[i].layer_output_r, layers[i].pool_flag_r, layers[i].field_width_out,
												layers[i].field_height_out,	layers[i].layer_depth_out, batch_size);
	}
	else
		layers[i].layer_output = layers[i].temp;

	if (verbose)
	{
		std::cout<<"\tPooled\n";
		std::cout<<"done with layer " << i <<"\n\n";
	}
	if (output)
	{
		std::cout<<"printing output from layer " << i <<std::endl;
		io::print_output(1,0,layers[i],"pooled_output");
		std::cout<<"printing weights from layer " << i <<std::endl;
		io::print_weights(layers[i]);
	}
}

void network::run_softpool()
{
	if (verbose)
		std::cout<<"Soft Pool\n";

	int i = layers.size()-1;
	int step_size = 10; //todo this is 10 for cifar 10
	unsigned int position;
	double max_val;

	for (int j=0; j<batch_size; j++)
	{
		thrust::device_vector<double>::iterator iter =
			thrust::max_element(layers[i].layer_output.begin()+step_size*j, layers[i].layer_output.begin()+step_size*(j+1));
		position = iter-layers[i].layer_output.begin();
		position = position%step_size;
		max_val = *iter;
		if (verbose)
			std::cout << max_val*100 << "% confident that image " << j << " is a " << io::CIFAR10_int_to_class(position) << std::endl;
	}

	if (verbose)
		std::cout<<"Soft Pool Done\n";
}
