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

namespace io
{
//number of images in the batch file
const int number_of_images = 10000;
//each image is 32x32x3
const int n_rows = 32;
const int n_cols = 32;

void read_batch(string filename, thrust::host_vector<int> &images, thrust::host_vector<int> &labels)
{
	//open batch file
    ifstream file(filename.c_str(), ios::binary);
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
    	cout<<"Failed to open file:" << filename.c_str() << endl;
}

void read_CIFAR10(image_DB &idb)
{
    string filename;

    idb.batch_1.resize(n_rows*n_cols*3 * number_of_images);
    idb.batch_1_labels.resize(number_of_images);
    filename = "cifar-10-batches-bin/data_batch_1.bin";
    read_batch(filename, idb.batch_1, idb.batch_1_labels);

    idb.training.resize(n_rows*n_cols*3 * number_of_images);
    idb.training_labels.resize(number_of_images);
    filename = "cifar-10-batches-bin/test_batch.bin";
    read_batch(filename, idb.training, idb.training_labels);
}

}//end namespace io

/*

 *
 */

