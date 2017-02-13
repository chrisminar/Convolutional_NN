/***************************************************************************//**
 * \file io.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the namespace io
 */

#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "image_DB.h"
using namespace std;

namespace io
{

void read_batch(string filename, thrust::host_vector<int> &vec, thrust::host_vector<int> &label);
void read_CIFAR10(image_DB &idb);


}//end namespace io
