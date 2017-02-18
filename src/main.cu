/***************************************************************************//**
 * \file main.cu
 * \author Christopher Minar (minarc@oregonstate.edu)
 * \brief Main source-file of \c convNet
 */

#include "io.h"
#include "types.h"
#include "image_DB.h"
#include "network.h"

int main()
{
	io::print_gpu_data();
	image_DB idb;
	network ntwrk(&idb);
    io::read_CIFAR10(idb);
	std::string fname = "/scratch/src/convNet/convNet/validation/CIFAR10.yaml";
	io::parse_network_file(fname, ntwrk);
	ntwrk.print_network_info();
	ntwrk.initialise_layers();

    return 0;
}

//todo make print info dump to a file
//todo figure out biases
//todo need to test each kernel and make a testing platform
//specifically need to test each of the indexes
//	todothis I need to create a method to print data from the kernels
