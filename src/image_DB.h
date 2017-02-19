/***************************************************************************//**
 * \file io.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementation of the class \c image_DB
 */

#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class image_DB
{
public:
	thrust::host_vector<double>	batch_1,
								batch_2,
								batch_3,
								batch_4,
								batch_5,
								training;

	thrust::host_vector<int>	batch_1_labels,
								batch_2_labels,
								batch_3_labels,
								batch_4_labels,
								batch_5_labels,
								training_labels;

	thrust::device_vector<double>	batch_1_D,
									batch_2_D,
									batch_3_D,
									batch_4_D,
									batch_5_D,
									training_D;

	thrust::device_vector<int>	batch_1_labels_D,
								batch_2_labels_D,
								batch_3_labels_D,
								batch_4_labels_D,
								batch_5_labels_D,
								training_labels_D;
};
