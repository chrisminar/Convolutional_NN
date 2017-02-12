/***************************************************************************//**
 * \file types.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the class layer
 */

#include <thrust/device_vector.h>
#include "types.h"

class layer
{
protected:
	//
	// Class Members
	//
	// Layer arrays
	thrust::device_vector<double> layer_input;
	thrust::device_vector<double> layer_output;
	thrust::device_vector<double> weights;

	// Layer metadata
	activation_function actv_fn;
	layer_type lyr_typ;
	layer_position lyr_pstn;
	layer *previous_layer;
	layer *next_layer;
	bool pool;

	// Layer hyperparameters
	int	field_width,
		field_height,
		stride_x,
		stride_y,
		zero_pad_x,
		zero_pad_y,
		filter_size,
		layer_depth;
	double learning_rate;



};
