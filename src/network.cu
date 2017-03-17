/***************************************************************************//**
 * \file network.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementaiton of the class \c network
 */

#include "io.h"
#include "network.h"
#include <vector>
#include <iostream>
#include <fstream>
#include "back_propogate/kernels/train.h"
#include <thrust/extrema.h>//keep
#include <thrust/functional.h>//keep
#include <thrust/transform.h>//keep

//constructor
network::network(image_DB *idb)
{
	IDB = idb;
	verbose = false;
	output = false;
	write = false;
}

void network::train_batch_1()
{
	int count = 0;
	while (count*batch_size < 10000)
	{
		std::cout << "Starting batch " << count+1 << "\n";
		//load new mini batch
		if (count > 0)
			load_batch(count);

		//train mini batch
		train_batch();

		//write weights
		write_weight_binary();

		count ++;
	}
}

void network::load_batch(int batch_num)
{
	int start_index = layers[0].field_width*layers[0].field_height*layers[0].layer_depth*batch_size*batch_num;
	mini_batch_r = thrust::raw_pointer_cast( &(IDB->batch_1_D[start_index]) );
	mini_batch_label_r = thrust::raw_pointer_cast( &(IDB->batch_1_labels_D[start_index]) );
	io::generate_target(target, IDB->batch_1_labels_D, batch_size, batch_num*batch_size, 10);
	layers[0].layer_input = mini_batch_r;
}

void network::train_batch()
{
	//training loop
	int count = 0;
	do
	{
		count ++;
		run();
		train_epoch();
		if (count % 10000 == 0)
		{
			std::cout << "L2 error at iteration " << count << " is " << error << "\t Break error is " << break_error << "\n";
			if (count >= 100000)
				{break;}
		}
	}
	while (!isdone());
	std::cout << "\tbatch trained in " << count << " iterations\n";
}

void network::train_epoch()
{
	//back propogate throught the nerual network
	for (int i=layers.size()-1; i>-1; i--) //loop from size-1 to 0
	{
		//resize and cast ddot
		layers[i].ddot_r = thrust::raw_pointer_cast( &(layers[i].ddot[0]) ); //todo this segment can probably be done without coping the output array

		//calculate deltas at current layer
		//delta = error at current layer * sigmoid of convoluted (layer input)
		//dE/dx = dE/dy * sigma'(x)
		//ddot = ddot * dsig_from_sig(temp)
		ddot_handler(i);

		//calculate weight gradiants at current layer
		//weight delta(delta, layer input)
		//dE/dw(dE/dx, y^l-1)
		//dw(ddot, layer_input)
		dw_handler(i);

		//propogate error up the network
		//previous layer delta = convolute(change in error from previous layer output, weights)
		//dE/dy^l-1 = convolute(dE/dx, weights)
		//ddot^l-1 = convolute(dE/d(convoluted input from this layer(not temp, but the thing between input and temp), weights)
		propogate_error_handler(i);
	}
}

//run the network
void network::run()
{
	int i;
	for (i=0; i < layers.size(); i++)
	{
		run_recast(i);
		run_convolute(i);
		run_activation(i);
		run_pool(i);
	}
	run_softpool();
}

bool network::isdone()
{
	int j = layers.size()-1;
	double threshold = 0.30;
	for (int i=0; i<target.size(); i++)
	{
		if ( abs(layers[j].layer_output[i] - target[i]) > threshold )
		{
			break_error = abs(layers[j].layer_output[i] - target[i]);
			return false;
		}
	}
	std::cout<< abs(layers[j].layer_output[0] - target[0]) << " " << layers[j].layer_output[0] << " " << target[0] << "\n";
	return true;
}

void network::initialise_layers()
{
	//copy image dbh from host to device
	IDB->batch_1_D = IDB->batch_1;
	IDB->batch_1_labels_D = IDB->batch_1_labels;
	IDB->training_D = IDB->training;
	IDB->training_labels_D = IDB->training_labels;
	//create pointers
	mini_batch_r = thrust::raw_pointer_cast( &(IDB->batch_1_D[0]) );
	test_data_r = thrust::raw_pointer_cast( &(IDB->training_D[0]) );
	mini_batch_label_r = thrust::raw_pointer_cast( &(IDB->batch_1_labels_D[0]) );
	test_data_label_r = thrust::raw_pointer_cast( &(IDB->training_labels_D[0]) );

	//initialse targets
	target.resize(batch_size * 10);
	io::generate_target(target, IDB->batch_1_labels_D, batch_size, 0, 10);

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
		//check if layer sizing is correct
		/*int outsize = (layers[i].field_height-layers[i].filter_size+layers[i].zero_pad_x*2)/layers[i].stride_x + 1;
		std::cout << "The convolution of layer " << layers[i].layer_position << " (size " << layers[i].field_height <<
				") will produce output of size " << outsize <<"\n\n";*/
	}
	//set next layer
	for (int i=0; i<layers.size(); i++)
	{
		if (i != layers.size() - 1)
			layers[i].next_layer = &layers[i+1];
		if (output and verbose)
			layers[i].print_metadata();
	}

	//print input
	if (output and verbose)
	{
		io::print_input(1, layers[0].field_width, layers[0].field_height, layers[0].layer_depth, IDB->batch_1_D, "input");
	}
}

void network::write_weight_binary()
{
	std::ofstream myfile;
	myfile.open("weights", std::ios::binary);
	if (myfile.is_open())
	{
		char buffer;
		for (int i=0; i<layers.size(); i++)
		{
			for (int j=0; j<layers[i].weights.size(); j++)
			{
				buffer = char(layers[i].weights[j]);
				myfile.write(&buffer, sizeof(double));
			}
		}
		myfile.close();
	}
	else
		std::cout << "Unable to open weight binary file\n";
}

void network::read_weight_binary()
{
	std::ifstream myfile;
	myfile.open("weights", std::ios::binary);
	if (myfile.is_open())
	{
		char buffer;
		for (int i=0; i<layers.size(); i++)
		{
			for (int j=0; j<layers[i].weights.size(); j++)
			{
				myfile.read(&buffer, sizeof(double));
				layers[i].weights[j] = double(buffer);
			}
		}
		myfile.close();
	}
	else
		std::cout << "Unable to open weight binary file\n";
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
	std::cout << "learning_rate: " << learning_rate << "\n\n";
}
