/***************************************************************************//**
 * \file network.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the class network
 */

#pragma once
#include <thrust/device_vector.h>
#include "types.h"

class database
{
public:
	std::vector<activation_function> activation_functions;
	std::vector<layer_connectivity> layer_connectivities;
	std::vector<bool> pools;
	thrust::device_vector<double>	batch_data,
									training_data;
	int	field_width,
		field_height,
		stride_x,
		stride_y,
		zero_pad_x,
		zero_pad_y,
		filter_size;
	double learning_rate;
};
