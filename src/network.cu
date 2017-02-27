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
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/merge.h>
#include <thrust/copy.h>

//constructor
network::network(image_DB *idb)
{
	IDB = idb;
}

void network::train_epoch()
{
	//step backwards through layers calculating deltas
	for (int i=layers.size()-1; i>-1; i--) //loop from size-1 to 0
	{
		//resize and cast delta
		layers[i].ddot_r = thrust::raw_pointer_cast( &(layers[i].dot_r[0]) );  //todo this segment can probably be done without coping the output array
		//if its the last layer
		if (i == layers.size()-1)
		{
			ddot_fc_layer(i);
		}
		else
		{
			ddot_conv_layer(i);
		}
	}
	//step forward through layers
	for (int i=0; i<layers.size(); i++)
	{

		//if fc   dweight = sum(layer_intput*ddot)
		if (i==layers[i].lyr_conv == FULLY_CONNECTED)
		{
			dw_fc_layer(i);
		}
		else
		{
			dw_conv_layer(i);
		}
		//dweight = dweight*trainingrate
		thrust::transform(layers[i].dweight.begin(), layers[i].dweight.end(), thrust::multiplies<double>());
		//weights -= trainingRate*dweight
		thrust::tansform(layers[i].weights.begin(), layers[i].weights.end(), layers[i].dweights.begin(), thrust::minus<double>());
		//bias something or other
	}
}

void network::ddot_fc_layer(int i)
{
	//note, can't handle FC with pooling
	//note I think this can only handle the last layer, possible not all FC layers
	//output_delta = layer_output - target
	thrust::transform(layers[i].layer_output.begin(), layers[i].layer_output.end(), //iterate over layer_output
						target.begin(), 											//subtract target
						layers[i].ddot.begin(), 									//place results in ddot
						thrust::minus<double>());									//subtract operator

	//Calculate L2 error: sum(delta^2)
	kernels::square<double> unary_op;
	thrust::plus<float> binary_op;
	error = thrust::transform_reduce(layers[i].ddot.begin(), layers[i].ddot.end(),	//iterate over ddot
									unary_op,										//square ddot
									0.0,											//start sum at 0
									binary_op);										//reduction sum

	//apply sigmoid' to ddot
	kernels::d_sig<double> dsig_op;
	thrust::transform(layers[i].ddot.begin(), layers[i].ddot.end(),					//iterate over ddot
						dsig_op);													//take the sigmoid derivative of the input
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

//rework, needs bias
void network::dw_conv_layer(int i)
{
	const int blocksize = 256;
	dim3 grid( int((layers[i].filter_size*layers[i].filter_size*layers[i]*layer_depth*layers[i].layer_depth_out)/blocksize)+1, 1);
	dim3 block(blocksize,1);
	kernels::calculate_dweight<<<grid,block>>>(layers[i].dweight_r, layers[i].temp_r, layers[i].dtemp_r, layers[i].filter_size, layers[i].field_height, layers[i].field_width,
												layers[i].layer_depth, layers[i].layer_depth_out, batch_size); //todo I think this should be operating on the layer input, not the temp
}

//done, needs bias
void network::dw_fc_layer(int i)
{
	const int blocksize = 256;
	dim3 grid( int(layers[i].weights.size()/blocksize)+1, 1);
	dim3 block(blocksize,1);
	kernels::calculate_fc_dweight<<<grid,block>>>(layers[i].dweight, layers[i].input, layers[i].dinput, layers[i].filter_size, layers[i].field_height, layers[i].field_width,
													layers[i].layer_depth, layers[i].layer_depth_out, batch_size);
}

//run the network
void network::run()
{
	//todo whats going on with weight deltas and deltas?
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
