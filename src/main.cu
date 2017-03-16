/***************************************************************************//**
 * \file main.cu
 * \author Christopher Minar (minarc@oregonstate.edu)
 * \brief Main source-file of \c convNet
 */

#include "io.h"
#include "types.h"
#include "image_DB.h"
#include "network.h"

int main(int argc, char **argv)
{
	//Reset device memory
	cudaDeviceReset();

	//Initialise image database and network
	image_DB idb;
	network ntwrk(&idb);
	io::command_line_parse(argc, argv, ntwrk);

	//Print information on gpu
	if (ntwrk.output and ntwrk.verbose)
		io::print_gpu_data();

	//read cifar 10 batch into idb
    io::read_CIFAR10(idb);

    //parse cifar 10 file to initialise the network
	std::string fname = "/scratch/src/convNet/convNet/validation/CIFAR10.yaml";
	io::parse_network_file(fname, ntwrk);
	//ntwrk.print_network_info();

	//initialise layers
	ntwrk.initialise_layers();

	//print device memory
	io::printDeviceMemoryUsage();

	//read weights
	//if (ntwrk.write)
	//	ntwrk.read_weight_binary();

	ntwrk.train_batch_1();

    return 0;
}

//todo make print info dump to a file

//todo biases
//todo save/read weights to file
//todo better momentum and training weight handleing
//todo multiple batches
//todo testing suite once thing has been run
//todo timing functions
//todo speedup
//todo make functions to test weights over the trianing data

//general questions:
//should weights be positive?
//what is the best way to introduce bias into our solution?
//why do we have to re cast arrays in network? --> why are the pointers changing between when they are first cast and run?
