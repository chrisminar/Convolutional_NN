/***************************************************************************//**
 * \file io.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementation of the class \c image_DB
 */

#pragma once

class image_DB
{
public:
	thrust::host_vector<int>	batch_1,
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
};
