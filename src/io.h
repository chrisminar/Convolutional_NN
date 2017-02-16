/***************************************************************************//**
 * \file io.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of the namespace io
 */

#pragma once
#include "image_DB.h"
#include "network.h"
#include "types.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace io
{

void read_batch(std::string filename, thrust::host_vector<int> &vec, thrust::host_vector<int> &label);
void read_CIFAR10(image_DB &idb);
void parse_network_file(std::string fname, network &N);
void print_gpu_data();

std::string layer_type_to_string(layer_type typ);
layer_type string_to_layer_type(std::string s);
std::string layer_connectivity_to_string(layer_connectivity conv);
layer_connectivity string_to_layer_connectivity(std::string s);
std::string activation_function_to_string(activation_function fn);
activation_function string_to_activation_function(std::string s);
bool string_to_bool(std::string s);

}//end namespace io
