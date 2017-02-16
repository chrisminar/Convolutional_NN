/***************************************************************************//**
 * \file network.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementaiton of the class \c network
 */

#include "io.h"
#include "network.h"
#include <vector>

//constructor
network::network(image_DB *idb)
{
	IDB = idb;
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

	//todo input/minibatch need to beinitilised and resized and whatnot before layers are initialised
	//make layers
	for (int i=0; i<activation_functions.size(); i++)
	{
		if (i == 0)
		{
			layers.push_back(layer(mini_batch_r, field_height, field_width, stride_x, stride_y, zero_pad_x, zero_pad_y, filter_size, 3));
			layers[0].lyr_typ = INPUT;
			layers[i].layer_depth_out = layer_depth[i+1];
		}
		else if (i == activation_functions.size() - 1)
		{
			layers.push_back(layer(layers[i-1].layer_output_r, &layers[i-1]));
			layers[i].lyr_typ = OUTPUT;
			layers[i].layer_depth_out = layer_depth[i];
		}
		else
		{
			layers.push_back(layer(layers[i-1].layer_output_r, &layers[i-1]));
			layers[i].lyr_typ = HIDDEN;
			layers[i].layer_depth_out = layer_depth[i+1];
		}
		layers[i].lyr_conv =  layer_connectivities[i];
		layers[i].pool = pools[i];
		layers[i].actv_fn = activation_functions[i];
		layers[i].layer_position = i;
		layers[i].filter_size = filter_size;
		layers[i].layer_depth = layer_depth[i];
	}
	//set next layer
	for (int i=0; i<layers.size(); i++)
	{
		if (i != layers.size() - 1)
			layers[i].next_layer = &layers[i+1];
		layers[i].print_metadata();
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
	std::cout << "\nlayer width: " << field_width << std::endl;
	std::cout << "layer height: " << field_height << std::endl;
	std::cout << "stride x: " << stride_x << std::endl;
	std::cout << "stride y: " << stride_y << std::endl;
	std::cout << "zero pad x: " << zero_pad_x << std::endl;
	std::cout << "zero pad y: " << zero_pad_y << std::endl;
	std::cout << "learning_rate: " << learning_rate << std::endl;
}
