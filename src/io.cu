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

#include "io.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

namespace io
{
//number of images in the batch file
const int number_of_images = 10000;
//each image is 32x32x3
const int n_rows = 32;
const int n_cols = 32;


void command_line_parse(int argc, char **argv, network &N)
{
	for (int i=1; i<argc; i++)
	{
		if (strcmp(argv[i],"-o")==0)
		{
			N.output = true;
		}
		if (strcmp(argv[i],"-v")==0)
			N.verbose = true;
	}
}

//read data from a cifar-10 file
void read_batch(std::string filename, thrust::host_vector<double> &images, thrust::host_vector<int> &labels)
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
						images[pixel_index] = double(int(pixel_buffer))/128;
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

std::string CIFAR10_int_to_class(unsigned int value)
{
	std::vector<std::string> classes(10);
	classes = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
	return classes[value];
}

void generate_target(thrust::device_vector<double> &target, thrust::device_vector<int> &label,
		int batch_size, int start_image, int num_class)
{
	double high_bar = 0.95;
	double low_bar = 0.05;
	int target_index = 0;
	for (int label_index=start_image; label_index< start_image+batch_size; label_index++)
	{
		for (int i=0; i<num_class; i++)
		{
			target_index = label_index*num_class + i;
			if (i == label[label_index])
			{
				target[target_index] = high_bar;
			}
			else
				target[target_index] = low_bar;
		}
	}
}
//read all the cifar-10 files
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

//handle the yaml read
void operator >> (const YAML::Node & node, network &N)
{
	node["batch_size"] >> N.batch_size;
	node["field_size"][0] >> N.field_width;
	node["field_size"][1] >> N.field_height;
	node["stride"][0] >> N.stride_x;
	node["stride"][1] >> N.stride_y;
	node["zero_pad"][0] >> N.zero_pad_x;
	node["zero_pad"][1] >> N.zero_pad_y;
	node["filter_size"] >> N.filter_size;
	node["learning_rate"] >> N.learning_rate;
	//layer depth
	const YAML::Node &depthi = node["depth_in"];
	const YAML::Node &deptho = node["depth_out"];
	int temp_depth;
	for (int i=0; i<depthi.size(); i++)
	{
		depthi[i] >> temp_depth;
		N.layer_depth.push_back(temp_depth);
		deptho[i] >> temp_depth;
		N.layer_depth_out.push_back(temp_depth);
	}
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

//parse the network
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

//print info about the gpu
void print_gpu_data()
{
    const int kb = 1024;
    const int mb = kb * kb;
    std::cout << "NBody.GPU" << std::endl << "=========" << std::endl << std::endl;

    std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;
    std::cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl << std::endl;

    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "CUDA Devices: " << std::endl << std::endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
        std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
        std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
        std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
        std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

        std::cout << "  Warp size:         " << props.warpSize << std::endl;
        std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
        std::cout << std::endl;
    }
}

//print the input to a layer to file
void print_input(int number_of_images, int fw, int fh, int ld, thrust::device_vector<double> input, std::string s)
{
	int pos = 0;
	//setup fstream
	std::ofstream myfile;
	std::string folder = "/scratch/src/convNet/convNet";
	std::stringstream out;
	std::stringstream convert; convert << "/output/" <<s<<":_image:"<<"0 - "<<number_of_images<<".csv";

	std::string folder_name = convert.str();
	out<<folder<<folder_name;
	myfile.open(out.str().c_str());
	for (int m=0; m < number_of_images; m++) //loop though images
	{
		myfile << "\nImage number: "<<m<<std::endl;
		for (int k=0; k<ld; k++) //loop through layers
		{
			myfile << "Layer number: "<<k<<std::endl;
			for (int j=0; j< fh; j++) //loop through rows
			{
				for (int i=0; i<fw; i++) //loop though cols
				{
					pos = m*fw*fh*ld + k*fw*fh + fw*j + i;
					myfile << input[pos] << ", ";
				}
				myfile << "\n";
			}
		}
	}
}

//print the temp of a layer to file
void print_temp(int number_of_images, int image_start, layer &layerin, std::string s)
{
	int pos = 0,
		fw,
		fh,
		ld;
	if (layerin.lyr_conv == FULLY_CONNECTED)
	{
		fw = layerin.field_width_out;
		fh = layerin.field_height_out;
		ld = layerin.layer_depth_out;
	}
	else
	{
		fw = layerin.field_width;
		fh = layerin.field_height;
		ld = layerin.layer_depth_out;
	}


	//setup fstream
	std::ofstream myfile;
	std::string folder = "/scratch/src/convNet/convNet";
	std::stringstream out;
	std::stringstream convert; convert << "/output/" <<s<<"_layer:"<<layerin.layer_position<< "_image:"<<image_start<<"-"<<image_start+number_of_images<<".csv";
	std::string folder_name = convert.str();
	out<<folder<<folder_name;
	myfile.open(out.str().c_str());
	for (int m=0; m < number_of_images; m++) //loop though images
	{
		myfile << "\nImage number: "<<m<<std::endl;
		//std::cout << "\nImage number: "<<m<<std::endl;
		for (int k=0; k<ld; k++) //loop through layers
		{
			myfile << "Layer number: "<<k<<std::endl;
			//std::cout << "Layer number: "<<k<<std::endl;
			for (int j=0; j< fh; j++) //loop through rows
			{
				for (int i=0; i<fw; i++) //loop though cols
				{
					pos = m*fw*fh*ld + k*fw*fh + fw*j + i;
					myfile << layerin.temp[pos] << ", ";
					//std::cout << pos << ", ";
				}
				myfile << "\n";
				//std::cout << "\n";
			}
		}
	}
}

//print the temp of a layer to file
void print_ddot(int number_of_images, int image_start, layer &layerin, std::string s)
{
	int pos = 0,
		fw,
		fh,
		ld;
	if (layerin.lyr_conv == FULLY_CONNECTED)
	{
		fw = layerin.field_width_out;
		fh = layerin.field_height_out;
		ld = layerin.layer_depth_out;
	}
	else
	{
		fw = layerin.field_width;
		fh = layerin.field_height;
		ld = layerin.layer_depth_out;
	}

	//setup fstream
	std::ofstream myfile;
	std::string folder = "/scratch/src/convNet/convNet";
	std::stringstream out;
	std::stringstream convert; convert << "/output/" <<s<<"_layer:"<<layerin.layer_position<< "_image:"<<"0-"<<number_of_images<<".csv";
	std::string folder_name = convert.str();
	out<<folder<<folder_name;
	myfile.open(out.str().c_str());
	for (int m=0; m < number_of_images; m++) //loop though images
	{
		myfile << "\nImage number: "<<m<<std::endl;
		for (int k=0; k<ld; k++) //loop through layers
		{
			myfile << "Layer number: "<<k<<std::endl;
			for (int j=0; j< fh; j++) //loop through rows
			{
				for (int i=0; i<fw; i++) //loop though cols
				{
					pos = m*fw*fh*ld + k*fw*fh + fw*j + i;
					myfile << layerin.ddot[pos] << ", ";
				}
				myfile << "\n";
			}
		}
	}
}

//print the layer weights to a file
void print_weights(layer &layerin, std::string s)
{
	int pos = 0,
		ldo = layerin.layer_depth_out,
		ld = layerin.layer_depth,
		fs;
	if (layerin.lyr_conv == FULLY_CONNECTED)
	{
		fs = layerin.field_width;
	}
	else
	{
		fs = layerin.filter_size;
	}

	//setup fstream
	std::ofstream myfile;
	std::string folder = "/scratch/src/convNet/convNet";
	std::stringstream out;
	std::stringstream convert; convert << "/output/"<<s<<"_layer:"<<layerin.layer_position<<".csv";
	std::string folder_name = convert.str();
	out<<folder<<folder_name;
	myfile.open(out.str().c_str());
	for (int m=0; m < ldo; m++) //loop though layer depth out
	{
		myfile << "\nLayer out number: "<<m<<std::endl;
		for (int k=0; k<ld; k++) //loop through layer depth in
		{
			myfile << "Layer in number: "<<k<<std::endl;
			for (int j=0; j< fs; j++) //loop through rows
			{
				for (int i=0; i<fs; i++) //loop though cols
				{
					pos = m*fs*fs*ld + k*fs*fs + fs*j + i;
					myfile << layerin.weights[pos] << ", ";
				}
				myfile << "\n";
			}
		}
	}
}

void print_dw(layer &layerin)
{
	int pos = 0,
		ldo = layerin.layer_depth_out,
		ld = layerin.layer_depth,
		fs;
	if (layerin.lyr_conv == FULLY_CONNECTED)
	{
		fs = layerin.field_width;
	}
	else
	{
		fs = layerin.filter_size;
	}

	//setup fstream
	std::ofstream myfile;
	std::string folder = "/scratch/src/convNet/convNet";
	std::stringstream out;
	std::stringstream convert; convert << "/output/dw"<<"_layer:"<<layerin.layer_position<<".csv";
	std::string folder_name = convert.str();
	out<<folder<<folder_name;
	myfile.open(out.str().c_str());
	for (int m=0; m < ldo; m++) //loop though layer depth out
	{
		myfile << "\nLayer out number: "<<m<<std::endl;
		for (int k=0; k<ld; k++) //loop through layer depth in
		{
			myfile << "Layer in number: "<<k<<std::endl;
			for (int j=0; j< fs; j++) //loop through rows
			{
				for (int i=0; i<fs; i++) //loop though cols
				{
					pos = m*fs*fs*ld + k*fs*fs + fs*j + i;
					myfile << layerin.dweight[pos] << ", ";
				}
				myfile << "\n";
			}
		}
	}
}

//print the output to a file
void print_output(int number_of_images, int image_start, layer &layerin, std::string s)
{
	int pos = 0,
		fw = layerin.field_width_out,
		fh = layerin.field_height_out,
		ld = layerin.layer_depth_out;

	//setup fstream
	std::ofstream myfile;
	std::string folder = "/scratch/src/convNet/convNet";
	std::stringstream out;
	std::stringstream convert; convert << "/output/"  <<s<<"_layer:"<<layerin.layer_position<< "_image:"<<image_start<<"-"<<image_start+number_of_images<<".csv";
	std::string folder_name = convert.str();
	out<<folder<<folder_name;
	myfile.open(out.str().c_str());
	for (int m=0; m < number_of_images; m++) //loop though images
	{
		myfile << "\nImage number: "<<m<<std::endl;
		for (int k=0; k<ld; k++) //loop through layers
		{
			myfile << "Layer number: "<<k<<std::endl;
			for (int j=0; j< fh; j++) //loop through rows
			{
				for (int i=0; i<fw; i++) //loop though cols
				{
					pos = m*fw*fh*ld + k*fw*fh + fw*j + i;
					myfile << layerin.layer_output[pos] << ", ";
				}
				myfile << "\n";
			}
		}
	}
}

//print the device memory usage
void printDeviceMemoryUsage()
{
	size_t _free, _total;
	cudaMemGetInfo(&_free, &_total);
	std::cout << '\n' << "Initialisation complete\nFlux capacitors charged" << ": Memory Usage " << (_total-_free)/(1024.0*1024*1024) \
	          << " / " << _total/(1024.0*1024*1024) << " GB" << '\n' << std::endl;
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
