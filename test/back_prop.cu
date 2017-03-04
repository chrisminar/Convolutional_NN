/***************************************************************************//**
 * \file back_prop.cu
 * \author Christopher Minar (minarc@oregonstate.edu)
 * \brief tests back propogation kernels
 */

#include "kernels/train_test.h"
#include "back_prop.h"

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
//#include <random>
//#include <vector>
//#include <algorithm> //std::any_of

#include <thrust/device_vector.h>
//#include <thrust/transform.h>
//#include <thrust/transform_reduce.h>
//#include <thrust/functional.h>
//#include <thrust/sequence.h>

//tests all the operator functions
void test_back_prop()
{
	test_calculate_dweight();
	test_calculate_dweight_fc();
}

bool test_calculate_dweight()
{
	//initialise arrays to test calculate dweight
	thrust::device_vector<double>	dweight,
									input,
									ddot,
									sum;
	thrust::device_vector<int>		pool_flag,
									weight_index,
									outer_layer_index,
									outer_layer_number,
									inner_layer_index,
									inner_layer_number,
									layer_x,
									layer_y,
									count,
									input_index,
									ddot_index;

	//init pointers
	double *dweight_r,
			*input_r,
			*ddot_r,
			*sum_r;
	int *pool_flag_r,
		*weight_index_r,
		*outer_layer_index_r,
		*outer_layer_number_r,
		*inner_layer_index_r,
		*inner_layer_number_r,
		*layer_x_r,
		*layer_y_r,
		*count_r,
		*input_index_r,
		*ddot_index_r;

	//initialise testing values
	int filter_size = 3,
		field_height = 4,
		field_width = 4,
		layer_depth = 2,
		layer_depth_out = 3,
		batch_size = 2;

	//resize arrays
	dweight.resize(filter_size*filter_size*layer_depth*layer_depth_out);
	input.resize(field_width*field_height*layer_depth*batch_size);
	ddot.resize(field_width*field_height*layer_depth_out*batch_size);
	pool_flag.resize(5); //not used right now
	weight_index.resize(dweight.size());
	outer_layer_index.resize(dweight.size());
	outer_layer_number.resize(dweight.size());
	inner_layer_index.resize(dweight.size());
	inner_layer_number.resize(dweight.size());
	layer_x.resize(dweight.size());
	layer_y.resize(dweight.size());
	sum.resize(dweight.size());
	count.resize(dweight.size());
	input_index.resize(dweight.size()*batch_size*field_height*field_width);
	ddot_index.resize(dweight.size()*batch_size*field_height*field_width);

	//cast points
	dweight_r = thrust::raw_pointer_cast( &(dweight[0]) );
	input_r = thrust::raw_pointer_cast( &(input[0]) );
	ddot_r = thrust::raw_pointer_cast( &(ddot[0]) );
	pool_flag_r = thrust::raw_pointer_cast( &(pool_flag[0]) );
	weight_index_r = thrust::raw_pointer_cast( &(weight_index[0]) );
	outer_layer_index_r = thrust::raw_pointer_cast( &(outer_layer_index[0]) );
	outer_layer_number_r = thrust::raw_pointer_cast( &(outer_layer_number[0]) );
	inner_layer_index_r = thrust::raw_pointer_cast( &(inner_layer_index[0]) );
	inner_layer_number_r = thrust::raw_pointer_cast( &(inner_layer_number[0]) );
	layer_x_r = thrust::raw_pointer_cast( &(layer_x[0]) );
	layer_y_r = thrust::raw_pointer_cast( &(layer_y[0]) );
	sum_r = thrust::raw_pointer_cast( &(sum[0]) );
	count_r = thrust::raw_pointer_cast( &(count[0]) );
	input_index_r = thrust::raw_pointer_cast( &(input_index[0]) );
	ddot_index_r = thrust::raw_pointer_cast( &(ddot_index[0]) );

	//call kernel
	const int blocksize = 256;
	dim3 grid( int((filter_size*filter_size*layer_depth*layer_depth_out)/blocksize)+1, 1);
	dim3 block(blocksize,1);
	kernels::calculate_dweight_test<<<grid,block>>>(dweight_r, input_r, ddot_r, pool_flag_r,
													filter_size, field_height, field_width,
													layer_depth, layer_depth_out, batch_size,
													weight_index_r, outer_layer_index_r, outer_layer_number_r,
													inner_layer_index_r, inner_layer_number_r,
													layer_y_r, layer_x_r,
													count_r, sum_r, input_index_r, ddot_index_r);

	//prints to /scratch/src/convNet/convNet/test/output
	print_weights(layer_depth_out, layer_depth, filter_size, "weight_index", weight_index);
	print_weights(layer_depth_out, layer_depth, filter_size, "outer_layer_index", outer_layer_index);
	print_weights(layer_depth_out, layer_depth, filter_size, "outer_layer_number", outer_layer_number);
	print_weights(layer_depth_out, layer_depth, filter_size, "inner_layer_number", inner_layer_number);
	print_weights(layer_depth_out, layer_depth, filter_size, "inner_layer_index", inner_layer_index);
	print_weights(layer_depth_out, layer_depth, filter_size, "layer_x", layer_x);
	print_weights(layer_depth_out, layer_depth, filter_size, "layer_y", layer_y);
	//indexing looks good
	print_weights(layer_depth_out, layer_depth, filter_size, "count", count);
	print_weights(layer_depth_out, layer_depth, filter_size, "sum", sum);
	print_dweight_test_loop(layer_depth_out, layer_depth, filter_size, batch_size,
								field_width, field_height, "input_index", input_index);
	print_dweight_test_loop(layer_depth_out, layer_depth, filter_size, batch_size,
								field_width, field_height, "ddot_index", ddot_index);
	//loop indexing looks good too

	return true;
}

//YOU ARE HERE
bool test_calculate_dweight_fc()
{
	return true;
}


template<typename T>
void print_weights(int ldo, int ld, int fs, std::string s, thrust::device_vector<T> w)
{
	int pos = 0;
	//setup fstream
	std::ofstream myfile;
	std::string folder = "/scratch/src/convNet/convNet/test/";
	std::stringstream out;
	std::stringstream convert; convert << "/output/weights_" <<s<< ".csv";
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
					myfile << w[pos] << ", ";
				}
				myfile << "\n";
			}
		}
	}
}

void print_dweight_test_loop(int ldo, int ld, int fs, int bs, int fw, int fh, std::string s, thrust::device_vector<int> w)
{
	int pos = 0;
	//setup fstream
	std::ofstream myfile;
	std::string folder = "/scratch/src/convNet/convNet/test/";
	std::stringstream out;
	std::stringstream convert; convert << "/output/weights_" <<s<< ".csv";
	std::string folder_name = convert.str();
	out<<folder<<folder_name;
	myfile.open(out.str().c_str());
	for (int m=0; m < ldo; m++) //loop though layer depth out
	{
		myfile << "\nLayer out number: "<<m<<std::endl;
		for (int k=0; k<ld; k++) //loop through layer depth in
		{
			myfile << "Layer in number: "<<k<<std::endl;
			//loop through filter rows
			for (int j=0; j< fs; j++)
			{
				//loop through field rows
				for (int y=0; y<fh; y++)
				{
					//loop through images
					for (int n=0; n<bs; n++)
					{
						//loop though filter cols
						for (int i=0; i<fs; i++)
						{
							//loop through field cols
							for (int x=0; x<fw; x++)
							{
								pos = (m*fs*fs*ld + k*fs*fs + fs*j + i) * bs*fh*fw;
								pos += n*fw*fw + y*fw + x;
								myfile << w[pos] << ", ";
							}//end field cols
							myfile << "\t";
						}//end filter cols
						myfile << "\t";
					}//end images
					myfile << "\n";
				}//end field row
				myfile << "\n";
			}//end filter row
			myfile << "\n";
		}//end layer depth in
	}//end layer depth out
}
