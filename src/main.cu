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

	//training loop
	int count = 0;
	do
	{
		count ++;
		ntwrk.run();
		ntwrk.train_epoch();
		std::cout << "L2 error at iteration " << count << " is " << ntwrk.error << "\n";
		if (count > 100000)
			break;
		if (count % 10000 == 0)
		{
			for (int j=0;j<10;j++)
				std::cout<<ntwrk.target[j]<< " "<<ntwrk.layers[3].layer_output[j]<<"\n";
		}
	}
	//while (count < 1);
	while (!ntwrk.isdone());

    return 0;
}

//todo make print info dump to a file
//todo set up verbosity in argparse

//todo biases
//todo save/read weights to file
//todo better momentum and training weight handleing
//todo multiple batches
//todo testing suite once thing has been run
//todo timing functions
//todo speedup

//general questions:
//should weights be positive?
//what is the best way to introduce bias into our solution?
//why do we have to re cast arrays in network? --> why are the pointers changing between when they are first cast and run?
