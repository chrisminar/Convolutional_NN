/***************************************************************************//**
 * \file network.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the class network
 */

#pragma once
#include "io.h"
#include "layer.h"
#include "image_DB.h"
#include <thrust/device_vector.h>
#include <vector>

class network
{
public:
	image_DB *IDB;

	std::vector<layer> layers;										//array of all the layers in the network
	std::vector<activation_function> activation_functions;			//the activation function to use for each layer
	std::vector<layer_connectivity> layer_connectivities;			//is the layer convolutional or fully connected?
	std::vector<bool> pools;										//should the layer be pooled at the end?
	std::vector<int> layer_depth;									//how deep is each layer?

	int		*mini_batch_r,											//pointer to a segment of the training data
			*test_data_r,											//pointer to test data
			*mini_batch_label_r,
			*test_data_label_r;

	int	field_width,												//layer width
		field_height,												//layer height
		stride_x,													//layer x stride
		stride_y,													//layer y stride
		zero_pad_x,													//layer x padding
		zero_pad_y,													//layer y padding
		filter_size,												//the size of the convolutional filter to use
		batch_size;													//the number of images to train on
	double learning_rate;

	//constructor
	network(image_DB *idb);
	void initialise_layers();
	void print_network_info();
	void parse_network_file(std::string fname);

};
