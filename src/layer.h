/***************************************************************************//**
 * \file types.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the class layer
 */

#pragma once
#include <thrust/device_vector.h>
#include "types.h"
#include "io.h"

class layer
{
public:
	//
	// Class Members
	//
	// Layer arrays
	double *layer_input;
	thrust::device_vector<double> layer_output;
	thrust::device_vector<double> weights;

	// Layer metadata
	activation_function actv_fn;
	layer_connectivity lyr_conv;
	layer_type lyr_typ;
	layer *previous_layer;
	layer *next_layer;
	bool pool;

	// Layer hyperparameters
	int	field_width,
		field_height,
		field_width_out,
		field_height_out,
		stride_x,
		stride_y,
		zero_pad_x,
		zero_pad_y,
		filter_size,
		layer_depth, //todo setup layer_depth out
		layer_depth_out,
		layer_position;
	double learning_rate;

	//array pointers
	//double	*layer_output_r,
	//		*weights_r;

	//
	// Class Methods
	//
	// Constructors
	layer(double *layer_input, layer *prev_layer);
	layer(double *layer_input, int field_size, int stride, int zero_pad, int fltr_size, int lyr_depth);
	layer(double *layer_input, int field_width, int field_height, int stride_x, int stride_y, int zero_pad_x,
			int zero_pad_y, int filter_size, int lyr_depth);
	void initialise();
	void set_pool(bool pool_);
	void set_next_layer(layer *nxt_layer);
	void set_layer_connectivity(layer_connectivity layer_connectivity_);
	void set_layer_position(int layer_position);
	void set_layer_type(layer_type lyr_typ_);
	void set_activation_function(activation_function actv_fn_);
	void print_metadata();
};
