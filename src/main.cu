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
	//Reset device memory
	cudaDeviceReset();

	//Print information on gpu
	//io::print_gpu_data();

	//Initialise image database and network
	image_DB idb;
	network ntwrk(&idb);

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

	//training loop
	int count = 0;
	do
	{
		count ++;
		ntwrk.run();
		ntwrk.train_epoch();
		std::cout << "L2 error at iteration " << count << " is " << ntwrk.error << "\n";
		if (count > 10000)
			break;
	}
	while (!ntwrk.isdone());

    return 0;
}

//todo make print info dump to a file
//todo save/read weights to file

//todo write outputs for backpropogation
//todo biases
//todo timing functions

//general questions:
//should weights be positive?
//what is the best way to introduce bias into our solution?
//why do we have to re cast arrays in network? --> why are the pointers changing between when they are first cast and run?
