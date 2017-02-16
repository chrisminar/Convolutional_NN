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
//todo initialise weights
//todo write activation function
//todo write pool
//todo figure out biases
//you are currently working on implementing backpropogation in layers