/***************************************************************************//**
 * \file io.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the namespace io
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace io
{
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


void read_batch(string filename, thrust::host_vector &vec, thrust::host_vector &label);


}//end namespace io
