/***************************************************************************//**
 * \file io.cpp
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementation of the namespace io
 * // readCIFAR10.cc
 * \
 * \feel free to use this code for ANY purpose
 * \author : Eric Yuan
 * \my blog: http://eric-yuan.me/
 */

#include <fstream>
#include <iostream>
#include "io.h"
#include <vector>
#include <thrust/host_vector.h>
#include <yaml-cpp/yaml.h>

namespace io
{
//number of images in the batch file
const int number_of_images = 10000;
//each image is 32x32x3
const int n_rows = 32;
const int n_cols = 32;

void read_batch(std::string filename, thrust::host_vector<int> &images, thrust::host_vector<int> &labels)
{
	//open batch file
    std::ifstream file(filename.c_str(), std::ios::binary);
    //if we sucessfully opened the file...
    if (file.is_open())
    {
		char cifar_10_label;
		char pixel_buffer;
		int pixel_index;
		//loop through every image in the batch
		for(int i = 0; i < number_of_images; ++i)
		{
			//read the data label
			file.read(&cifar_10_label, 1);
			labels[i] = int(cifar_10_label);

			//read image
			//loop through color channels
			for (int ch=0; ch<3; ++ch)
			{
				//loop through rows
				for (int r=0; r<n_rows; ++r)
				{
					//loop through columns
					for (int c=0; c<n_cols; ++c)
					{
						file.read(&pixel_buffer, 1);
						//				   previous images		  previous color channels   prev rows	prev columns
						pixel_index = (i * n_rows*n_cols*3 + 1) + (ch * n_rows * n_cols) + (r*n_rows) + c;
						images[pixel_index] = int(pixel_buffer);
					}
				}
			}
			//output should be [red], [green], [blue]
			//color = [row 1, ..., row n]
			//row = [col1, ..., coln]
		}
    }
    else
    	std::cout<<"Failed to open file:" << filename.c_str() << std::endl;
}

void read_CIFAR10(image_DB &idb)
{
    std::string filename;

    idb.batch_1.resize(n_rows*n_cols*3 * number_of_images);
    idb.batch_1_labels.resize(number_of_images);
    filename = "/scratch/src/convNet/convNet/cifar-10-batches-bin/data_batch_1.bin";
    read_batch(filename, idb.batch_1, idb.batch_1_labels);

    idb.training.resize(n_rows*n_cols*3 * number_of_images);
    idb.training_labels.resize(number_of_images);
    filename = "/scratch/src/convNet/convNet/cifar-10-batches-bin/test_batch.bin";
    read_batch(filename, idb.training, idb.training_labels);
}

void operator >> (const YAML::Node & node, network &N)
{
	node["field_size"][0] >> N.field_width;
	node["field_size"][1] >> N.field_height;
	node["stride"][0] >> N.stride_x;
	node["stride"][1] >> N.stride_y;
	node["zero_pad"][0] >> N.zero_pad_x;
	node["zero_pad"][1] >> N.zero_pad_y;
	node["learning_rate"] >> N.learning_rate;
	//activation functions
	const YAML::Node &actv = node["activation_functions"];
	std::string temp_actv;
	for (int i=0; i<actv.size(); i++)
	{
		actv[i] >> temp_actv;
		N.activation_functions.push_back(io::string_to_activation_function(temp_actv));
	}
	//pools
	const YAML::Node &pl = node["pools"];
	bool temp_pool;
	for (int i=0; i<pl.size(); i++)
	{
		pl[i] >> temp_pool;
		N.pools.push_back(temp_pool);
	}
	// layer connectivity
	const YAML::Node &conn = node["layer_connectivities"];
	std::string temp_conn;
	for (int i=0; i<conn.size(); i++)
	{
		conn[i] >> temp_conn;
		N.layer_connectivities.push_back(io::string_to_layer_connectivity(temp_conn));
	}
}

void parse_network_file(std::string fname, network &N)
{
	std::cout<<"Parsing network file " << fname << std::endl;
	std::ifstream fin(fname.c_str());
	YAML::Parser  parser(fin);
	YAML::Node    doc;
	parser.GetNextDocument(doc);
	for (unsigned i=0; i<doc.size(); i++)
	{
		doc[i] >> N;
	}
}

std::string layer_type_to_string(layer_type typ)
{
	if (typ == INPUT)
		return "INPUT";
	else if (typ == OUTPUT)
		return "OUTPUT";
	else if (typ == HIDDEN)
		return "HIDDEN";
	else
		return "NONE";
}
layer_type string_to_layer_type(std::string s)
{
	if (s == "INPUT")
		return INPUT;
	else if (s == "OUTPUT")
		return OUTPUT;
	else if (s == "HIDDEN")
		return HIDDEN;
	else
		return HIDDEN;
}
std::string layer_connectivity_to_string(layer_connectivity conv)
{
	if (conv == CONVOLUTIONAL)
		return "CONVOLUTIONAL";
	else if (conv == FULLY_CONNECTED)
		return "FULLY_CONNECTED";
	else
		return "NONE";
}
layer_connectivity string_to_layer_connectivity(std::string s)
{
	if (s == "CONVOLUTIONAL")
		return CONVOLUTIONAL;
	else if (s == "FULLY_CONNECTED")
		return FULLY_CONNECTED;
	else
		return CONVOLUTIONAL;
}
std::string activation_function_to_string(activation_function fn)
{
	if (fn == SIGMOID)
		return "SIGMOID";
	else if (fn == TANH)
		return "TANH";
	else if (fn == RELU)
		return "RELU";
	else
		return "NONE";
}
activation_function string_to_activation_function(std::string s)
{
	if (s == "SIGMOID")
		return SIGMOID;
	else if (s == "TANH")
		return TANH;
	else if (s == "RELU")
		return RELU;
	else
		return SIGMOID;
}
bool string_to_bool(std::string s)
{
	if (s == "true" || s == "True" || s == "TRUE")
		return true;
	else if (s == "false" || s == "False" || s == "FALSE")
		return false;
	else
		return false;
}
}//end namespace io


/*

 *
 */

