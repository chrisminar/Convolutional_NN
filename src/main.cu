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
	cudaDeviceReset();
	//io::print_gpu_data();
	image_DB idb;
	network ntwrk(&idb);
    io::read_CIFAR10(idb);
	std::string fname = "/scratch/src/convNet/convNet/validation/CIFAR10.yaml";
	io::parse_network_file(fname, ntwrk);
	//ntwrk.print_network_info();
	ntwrk.initialise_layers();
	io::printDeviceMemoryUsage();
	ntwrk.run();
	ntwrk.train_epoch();

    return 0;
}

//todo make print info dump to a file
//todo backprop

//todo make sure dw conv and dw fc are operating according to the new method, they should be close
//todo write the kernels for the ddot back propagation stuff
//todo move thrust headers between network.cu and back_propogate/train.cu
//todo biases


//general questions:
//should weights be positive?
//what is the best way to introduce bias into our solution?
//why do we have to re cast arrays in network? --> why are the pointers changing between when they are first cast and run?
