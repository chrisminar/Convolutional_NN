/***************************************************************************//**
 * \file layer.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementaiton of the class \c layer
 */

#include "layer.h"
#include "io.h"
#include <random>

void layer::initialise()
{
	//set output metadata
	if (pool)
	{
		field_width_out = field_width/2;
		field_height_out = field_height/2;
	}
	else
	{
		field_width_out = field_width;
		field_height_out = field_height;
	}
	//resize layer output
	layer_output.resize(field_width_out*field_height_out*layer_depth_out*batch_size);
	layer_output_r = thrust::raw_pointer_cast( &(layer_output[0]) );//this cast doesn't seem to work for some reason
	//resize temp
	temp.resize(field_width * field_height * layer_depth_out * batch_size);
	temp_r = thrust::raw_pointer_cast( &(temp[0]) );
	//give weights gaussian distribution
	thrust::host_vector<double> H;
	H.resize(filter_size*filter_size*layer_depth*layer_depth_out); //todo need some sort of bias term
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);
	double num = 0.0;
	int n = filter_size*filter_size*layer_depth;
	for (int i=0; i < H.size(); i++)
	{
		num = distribution(generator);
		H[i] = num/sqrt(n);
	}

	weights = H;
	weights_r = thrust::raw_pointer_cast ( &(weights[0]) );
}

// constructors
layer::layer(double *layer_input_, layer *previous_layer_)
{
	layer_input = layer_input_;
	previous_layer = previous_layer_;
	field_width = previous_layer->field_width_out;
	field_height = previous_layer->field_height_out;
	stride_x = previous_layer->stride_x;
	stride_y = previous_layer->stride_y;
	filter_size = previous_layer->filter_size;
	layer_depth = previous_layer->layer_depth;
	actv_fn = SIGMOID;
}

layer::layer(double *layer_input_, int field_size, int stride, int zero_pad, int filter_size_, int layer_depth_)
{
	layer_input = layer_input_;
	field_width = field_size;
	field_height = field_size;
	stride_x = stride;
	stride_y = stride;
	zero_pad_x = zero_pad;
	zero_pad_y = zero_pad;
	filter_size = filter_size_;
	layer_depth = layer_depth_;
	actv_fn = SIGMOID;
}

layer::layer(double *layer_input_, int field_width_, int field_height_, int stride_x_, int stride_y_, int zero_pad_x_,
		int zero_pad_y_, int filter_size_, int layer_depth_)
{
	layer_input = layer_input_;
	field_width = field_width_;
	field_height = field_height_;
	stride_x = stride_x_;
	stride_y = stride_y_;
	zero_pad_x = zero_pad_x_;
	zero_pad_y = zero_pad_y_;
	filter_size = filter_size_;
	layer_depth = layer_depth_;
	actv_fn = SIGMOID;
}

void layer::set_pool(bool pool_)
{
	pool = pool_;
}

void layer::set_next_layer(layer *next_layer_)
{
	next_layer = next_layer_;
}

void layer::set_layer_position(int layer_position_)
{
	layer_position = layer_position_;
}

void layer::set_layer_type(layer_type layer_type_)
{
	lyr_typ = layer_type_;
	if (layer_type_ == 0)
		lyr_typ = INPUT;
	else
		lyr_typ = HIDDEN;
}

void layer::set_layer_connectivity(layer_connectivity lyr_conv_)
{
	lyr_conv = lyr_conv_;
}

void layer::set_activation_function(activation_function actv_fn_)
{
	actv_fn = actv_fn_;
}

void layer::print_metadata()
{
	std::cout << "\n\nPrinting information for layer " << layer_position << std::endl;
	std::cout << "---------------------------------------------------------------------\n";
	std::cout << "Layer Type: " << io::layer_type_to_string(lyr_typ) << std::endl;
	std::cout << "Connectivity: " << io::layer_connectivity_to_string(lyr_conv) << std::endl;
	std::cout << "Activation_Function: " << io::activation_function_to_string(actv_fn) << std::endl;
	std::cout << "Pool layer: " << pool << std::endl;
	std::cout << "layer width: " << field_width << std::endl;
	std::cout << "layer height: " << field_height << std::endl;
	std::cout << "stride x: " << stride_x << std::endl;
	std::cout << "stride y: " << stride_y << std::endl;
	std::cout << "zero pad x: " << zero_pad_x << std::endl;
	std::cout << "zero pad y: " << zero_pad_y << std::endl;
	std::cout << "filter size: " << filter_size << std::endl;
	std::cout << "layer depth: " << layer_depth<< std::endl;
	std::cout << "layer_depth out: " << layer_depth_out << std::endl;
	std::cout << "learning_rate: " << learning_rate << std::endl;
}
