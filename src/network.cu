/***************************************************************************//**
 * \file network.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementaiton of the class \c network
 */

#include "io.h"
#include "network.h"
#include <vector>
#include "kernels/run.h"

//constructor
network::network(image_DB *idb)
{
	IDB = idb;
}

//run the network
void network::run()
{
	//todo whats going on with weight deltas and deltas?
	//todo check if inputs to convolute work, combination of zeropad
	const int blocksize = 256;

	//for (int i=0; i < layers.size(); i++)
	int i = 0;
	{
		//convolute each layer
		dim3 conv_g( int( (layers[i].field_width*layers[i].field_height*layers[i].layer_depth*batch_size - 0.5)/blocksize ) + 1, 1);
		dim3 conv_b(blocksize, 1);
		layers[i].temp_r = thrust::raw_pointer_cast( &(layers[i].temp[0]) );
		kernels::convolute<<<conv_g,conv_b>>>(layers[i].layer_input, layers[i].temp_r, layers[i].weights_r, layers[i].field_width,
											layers[i].field_height, layers[i].stride_x, layers[i].stride_y, layers[i].zero_pad_x,
											layers[i].zero_pad_y, layers[i].filter_size, batch_size, layers[i].layer_depth,
											layers[i].layer_depth_out);
		io::print_temp(1,0, layers[i], "before");
		//activation funciton on each layer
		dim3 actv_g( int( (layers[i].field_width*layers[i].field_height*layers[i].layer_depth_out*batch_size)/blocksize ) +1, 1);
		dim3 actv_b(blocksize,1);
		kernels::sigmoid_activation<<<actv_g, actv_b>>>(layers[i].temp_r, layers[i].field_width, layers[i].field_height,
														layers[i].layer_depth_out, layers[i].batch_size);
		std::cout<<"test temp out: " << layers[i].temp[5] <<std::endl;
		//pool layer
		dim3 pool_g ( int( (layers[i].field_width_out*layers[i].field_height_out*layers[i].layer_depth_out*batch_size)/blocksize ) +1, 1);
		dim3 pool_b (blocksize,1);
		layers[i].layer_output_r = thrust::raw_pointer_cast(&layers[i].layer_output[0]);
		kernels::pool_input<<<pool_g,pool_b>>>(layers[i].temp_r, layers[i].layer_output_r, layers[i].field_width_out,
										layers[i].field_height_out,	layers[i].layer_depth_out, batch_size);
		std::cout<<"test layer out: " << layers[i].layer_output[5] <<std::endl;
	}
	io::print_temp(1,0, layers[i], "after");
	io::print_weights(layers[i]);
	io::print_output(2,0,layers[i],"");
}

void network::initialise_layers()
{
	//copy image dbh from host to device
	IDB->batch_1_D = IDB->batch_1;
	IDB->batch_1_labels_D = IDB->batch_1_labels;
	IDB->training_D = IDB->training;
	IDB->training_labels_D = IDB->training_labels;
	//creat pointers
	mini_batch_r = thrust::raw_pointer_cast( &(IDB->batch_1_D[0]) );
	test_data_r = thrust::raw_pointer_cast( &(IDB->training_D[0]) );
	mini_batch_label_r = thrust::raw_pointer_cast( &(IDB->batch_1_labels_D[0]) );
	test_data_label_r = thrust::raw_pointer_cast( &(IDB->training_labels_D[0]) );
	//make layers
	for (int i=0; i<activation_functions.size(); i++)
	{
		if (i == 0)
		{
			layers.push_back(layer(mini_batch_r, field_height, field_width, stride_x, stride_y, zero_pad_x, zero_pad_y, filter_size, 3));
			layers[0].lyr_typ = INPUT;
		}
		else if (i == activation_functions.size() - 1)
		{
			layers.push_back(layer(layers[i-1].layer_output_r, &layers[i-1]));
			layers[i].lyr_typ = OUTPUT;
		}
		else
		{
			layers.push_back(layer(layers[i-1].layer_output_r, &layers[i-1]));
			layers[i].lyr_typ = HIDDEN;
		}
		layers[i].lyr_conv =  layer_connectivities[i];
		layers[i].pool = pools[i];
		layers[i].actv_fn = activation_functions[i];
		layers[i].layer_position = i;
		layers[i].filter_size = filter_size;
		layers[i].layer_depth = layer_depth[i];
		layers[i].layer_depth_out = layer_depth_out[i];
		layers[i].batch_size = batch_size;
		layers[i].learning_rate = learning_rate;
		layers[i].initialise();
	}
	//set next layer
	for (int i=0; i<layers.size(); i++)
	{
		if (i != layers.size() - 1)
			layers[i].next_layer = &layers[i+1];
		//layers[i].print_metadata();
	}
}

void network::print_network_info()
{
	std::cout << "\n\nPrinting network information\n";
	std::cout << "---------------------------------------------------------------------";
	std::cout << "\nConnectivity: ";
	for (int i=0; i<layer_connectivities.size(); i++)
		std::cout << io::layer_connectivity_to_string(layer_connectivities[i]) << "\t";
	std::cout << "\nActivation_Function: ";
	for (int i=0; i<activation_functions.size(); i++)
		std::cout << io::activation_function_to_string(activation_functions[i]) << "\t";
	std::cout << "\nPool layer: ";
	for (int i=0; i<pools.size(); i++)
			std::cout << pools[i] << "\t";
	std::cout << "\nLayer depths";
	std::cout << "\nin: ";
	for (int i=0; i<layer_depth.size(); i++)
			std::cout << layer_depth[i] << "\t";
	std::cout << "\nout: ";
	for (int i=0; i<layer_depth.size(); i++)
				std::cout << layer_depth_out[i] << "\t";
	std::cout << "\nlayer width: " << field_width << std::endl;
	std::cout << "layer height: " << field_height << std::endl;
	std::cout << "stride x: " << stride_x << std::endl;
	std::cout << "stride y: " << stride_y << std::endl;
	std::cout << "zero pad x: " << zero_pad_x << std::endl;
	std::cout << "zero pad y: " << zero_pad_y << std::endl;
	std::cout << "learning_rate: " << learning_rate << std::endl;
}
