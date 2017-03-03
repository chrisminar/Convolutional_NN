/***************************************************************************//**
 * \file types.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the class layer
 */

#pragma once
#include <thrust/device_vector.h>
#include "types.h"

class layer
{
public:
	//
	// Class Members
	//
	// Layer arrays
	thrust::device_vector<double> layer_output;
	thrust::device_vector<double> temp;
	thrust::device_vector<double> weights;
	thrust::device_vector<double> bias;
	thrust::device_vector<double> ddot;
	thrust::device_vector<double> ddot_temp;
	thrust::device_vector<double> dweight;
	thrust::device_vector<int> pool_flag;

	// layer array raw pointers
	double	*layer_input,
			*layer_output_r,
			*temp_r,
			*weights_r,
			*bias_r,
			*ddot_r,
			*ddot_temp_r,
			*dweight_r;
	int *pool_flag_r;

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
		layer_depth,
		layer_depth_out,
		layer_position,
		batch_size;
	double learning_rate;

	//array pointers
	//double	*layer_output_r,
	//		*weights_r;

	//
	// Class Methods
	//
	void initialise();
	// Constructors
	layer(double *layer_input, layer *prev_layer);
	layer(double *layer_input, int field_size, int stride, int zero_pad, int fltr_size, int lyr_depth);
	layer(double *layer_input, int field_width, int field_height, int stride_x, int stride_y, int zero_pad_x,
			int zero_pad_y, int filter_size, int lyr_depth);
	//io
	void print_metadata();
};
