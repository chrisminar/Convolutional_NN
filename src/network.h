/***************************************************************************//**
 * \file network.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the class network
 */

#pragma once
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
	std::vector<int> layer_depth;									//how deep is each layer coming in?
	std::vector<int> layer_depth_out;								//how deep is each layer going out?

	thrust::device_vector<double> target;
	thrust::device_vector<double> errorD;

	double	*mini_batch_r,											//pointer to a segment of the training data
			*test_data_r,											//pointer to test data
			*target_r;

	int *mini_batch_label_r,
		*test_data_label_r;

	int	field_width,												//layer width
		field_height,												//layer height
		stride_x,													//layer x stride
		stride_y,													//layer y stride
		zero_pad_x,													//layer x padding
		zero_pad_y,													//layer y padding
		filter_size,												//the size of the convolutional filter to use
		batch_size;													//the number of images to train on
	double learning_rate,
			error;

	//constructor
	network(image_DB *idb);

	//training
	void train_epoch();
	void propogate_error_handler(bool v, int i);
	void ddot_handler(bool v, int i);
	void dw_handler(bool v, int i);
	bool isdone();

	//back_propogate
	void initial_ddot(int i);
	void update_ddot(int i);
	void dw_conv(int i);
	void dw_fc(int i);
	void upstream_ddot_fc(int i);
	void upstream_ddot_conv(int i);

	//init
	void initialise_layers();

	//run
	void run();
	void run_recast(int i);
	void run_convolute(bool v, int i);
	void run_activation(bool v, int i);
	void run_pool(bool v, int i);
	void run_softpool(bool v);

	//io
	void print_network_info();
	void parse_network_file(std::string fname);


};
