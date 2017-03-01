/***************************************************************************//**
 * \file network.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementaiton of the class \c network
 */

#include "io.h"
#include "network.h"
#include <vector>
#include "kernels/run.h"
#include "kernels/train.h"
#include <thrust/extrema.h>//keep
#include <thrust/functional.h>//keep
#include <thrust/transform.h>//keep


//constructor
network::network(image_DB *idb)
{
	IDB = idb;
}

void network::train_epoch()
{
	//back propogate throught the nerual network
	for (int i=layers.size()-1; i>-1; i--) //loop from size-1 to 0
	{
		//resize and cast ddot
		layers[i].ddot_r = thrust::raw_pointer_cast( &(layers[i].dot_r[0]) );  //todo this segment can probably be done without coping the output array

		//propogate error up the network
		//previous layer delta = convolute(change in error from previous layer output, weights)
		//dE/dy^l-1 = convolute(dE/dx, weights)
		//ddot^l-1 = convolute(dE/d(convoluted input from this layer(not temp, but the thing between input and temp), weights)
		propogate_error_handler(i);

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
	}
}

/*
 * propogate error up the network
 * previous layer delta = convolute(change in error from previous layer output, weights)
 * dE/dy^l-1 = convolute(dE/dx, weights)
 * ddot^l-1 = convolute(dE/d(convoluted input from this layer(not temp, but the thing between input and temp), weights)
 */
void network::propogate_error_handler(int i)
{
	//if we are on a fully connected layer, upstream shouldn't be convoluted for
	if (layers[i].lyr_conv == FULLY_CONNECTED)
	{

	}
	//otherwise, call the convolution kernel
	else
	{

	}
}

/*
 * calculate deltas at current layer
 * delta = error at current layer * sigmoid of convoluted (layer input)
 * dE/dx = dE/dy * sigma'(x)
 * ddot = ddot * dsig_from_sig(temp)
 */
void network::ddot_handler(int i)
{
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
void network::dw_handler(int i)
{
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
	thrust::transform(layers[i].dweight.begin(), layers[i].dweight.end(), thrust::multiplies<double>());
	//update weights with weight deltas
	thrust::tansform(layers[i].weights.begin(), layers[i].weights.end(), layers[i].dweights.begin(), thrust::minus<double>());
}

//run the network
void network::run()
{
	const int blocksize = 256;

	int i;
	for (i=0; i < layers.size(); i++)
	{
		//resize stuff, not sure why this needs to be resized here but it breaks otherwise
		if (i>0)
		{
			layers[i].layer_input = thrust::raw_pointer_cast(&layers[i-1].layer_output[0]);
		}
		layers[i].temp_r = thrust::raw_pointer_cast( &(layers[i].temp[0]) );
		layers[i].layer_output_r = thrust::raw_pointer_cast( &(layers[i].layer_output[0]) );

		//setup grids and blocks
		dim3 actv_g( int( (layers[i].field_width*layers[i].field_height*layers[i].layer_depth_out*batch_size)/blocksize ) +1, 1);
		dim3 pool_g ( int( (layers[i].field_width_out*layers[i].field_height_out*layers[i].layer_depth_out*batch_size)/blocksize ) +1, 1);
		dim3 block (blocksize,1);
		//convolute each layer
		if (layers[i].lyr_conv == CONVOLUTIONAL)
		{
			dim3 conv_g( int( (layers[i].field_width*layers[i].field_height*layers[i].layer_depth*batch_size - 0.5)/blocksize ) + 1, 1);
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
		//std::cout<<"printing conv from layer " << i <<std::endl;
		//io::print_temp(1,0, layers[i], "conv");
		//activation function on each layer
		kernels::sigmoid_activation<<<actv_g, block>>>(layers[i].temp_r, layers[i].field_width, layers[i].field_height,
														layers[i].layer_depth_out, layers[i].batch_size);

		//pool layer
		if (layers[i].pool)
		{
			kernels::pool_input<<<pool_g,block>>>(layers[i].temp_r, layers[i].layer_output_r, layers[i].field_width_out,
													layers[i].field_height_out,	layers[i].layer_depth_out, batch_size);
		}
		else
			layers[i].layer_output = layers[i].temp;
		/*std::cout<<"printing sigm from layer " << i <<std::endl;
		io::print_temp(1,0, layers[i], "sigm");
		std::cout<<"printing weights from layer " << i <<std::endl;
		io::print_weights(layers[i]);
		std::cout<<"printing output from layer " << i <<std::endl;
		io::print_output(1,0,layers[i],"");
		std::cout<<"done with layer " << i <<"\n\n";*/
	}
	i = layers.size()-1;
	/*io::print_temp(2,0, layers[i], "after");
	io::print_weights(layers[i]);*/
	//io::print_output(3,0,layers[i],"");
	//soft pool
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
		//std::cout << max_val*100 << "% confident that image " << j << " is a " << io::CIFAR10_int_to_class(position) << std::endl;
	}
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

	//initialse first target
	batch_size = 1;
	target.resize(batch_size * 10);
	io::generate_target(target, IDB->batch_1_labels_D, batch_size, 0, 10);
	batch_size = 50;

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
		int outsize = (layers[i].field_height-layers[i].filter_size+layers[i].zero_pad_x*2)/layers[i].stride_x + 1;
		std::cout << "The convolution of layer " << layers[i].layer_position << " (size " << layers[i].field_height <<
				") will produce output of size " << outsize <<"\n\n";
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
