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
	if (lyr_typ == OUTPUT)
	{
		field_width_out = 1;
		field_height_out = 1;
	}

	//resize layer output
	layer_output.resize(field_width_out*field_height_out*layer_depth_out*batch_size);
	layer_output_r = thrust::raw_pointer_cast( &(layer_output[0]) );//this cast doesn't seem to work for some reason

	//resize temp
	if (lyr_typ != OUTPUT)
		temp.resize(field_width * field_height * layer_depth_out * batch_size);
	else
		temp.resize(field_width_out*field_height_out*layer_depth_out*batch_size);
	temp_r = thrust::raw_pointer_cast( &(temp[0]) );

	//resize ddot
	ddot.resize(field_width * field_height * layer_depth * batch_size);  //same size as input //ddot for output layer should be the size of the output
	ddot_temp.resize(ddot.size());
	ddot_r = thrust::raw_pointer_cast( & (ddot[0]) );
	ddot_temp_r = thrust::raw_pointer_cast( & (ddot_temp[0]) );

	//give weights and bias gaussian distribution
	thrust::host_vector<double> W;
	thrust::host_vector<double> B;
	std::default_random_engine generator(std::random_device{}());
	std::normal_distribution<double> distribution(0.0, 1.0);
	double num = 0.0;
	int n = 0;
	if (lyr_typ != OUTPUT)
	{
		W.resize(filter_size*filter_size*layer_depth*layer_depth_out);
		dweight.resize(filter_size*filter_size*layer_depth*layer_depth_out);
		n = filter_size*filter_size*layer_depth;
		for (int i=0; i < W.size(); i++)
		{
			W[i] = abs(num/n);
		}
	}
	else
	{
		W.resize(field_width*field_height*layer_depth*layer_depth_out);
		dweight.resize(field_width*field_height*layer_depth*layer_depth_out);
		n = field_width*field_height*layer_depth;
		for (int i=0; i<W.size(); i++)
		{
			num = distribution(generator);
			W[i] = abs(num/n);
		}
	}
	B.resize(layer_depth_out);
	for (int i=0; i<B.size(); i++)
	{
		num = distribution(generator);
		B[i] = abs(num/layer_depth_out);
	}

	weights = W;
	weights_r = thrust::raw_pointer_cast ( &(weights[0]) );
	thrust::fill(dweight.begin(), dweight.end(), 0.0);
	bias = B;
	bias_r = thrust::raw_pointer_cast ( &(bias[0]) );

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

void layer::print_metadata()
{
	std::cout << "\n\nPrinting information for layer " << layer_position << std::endl;
	std::cout << "---------------------------------------------------------------------\n";
	std::cout << "Layer Type: " << io::layer_type_to_string(lyr_typ) << std::endl;
	std::cout << "Connectivity: " << io::layer_connectivity_to_string(lyr_conv) << std::endl;
	std::cout << "Activation_Function: " << io::activation_function_to_string(actv_fn) << std::endl;
	std::cout << "Pool layer: " << pool << std::endl;
	std::cout << "layer width: " << field_width << "-" << field_width_out <<std::endl;
	std::cout << "layer height: " << field_height << "-" << field_height_out << std::endl;
	std::cout << "stride x: " << stride_x << std::endl;
	std::cout << "stride y: " << stride_y << std::endl;
	std::cout << "zero pad x: " << zero_pad_x << std::endl;
	std::cout << "zero pad y: " << zero_pad_y << std::endl;
	std::cout << "filter size: " << filter_size << std::endl;
	std::cout << "layer depth: " << layer_depth << "-" << layer_depth_out << std::endl;
	std::cout << "learning_rate: " << learning_rate << std::endl;
}
