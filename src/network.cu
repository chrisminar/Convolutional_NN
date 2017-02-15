/***************************************************************************//**
 * \file network.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementaiton of the class \c network
 */

#include "io.h"
#include "network.h"

//constructor
network::network(image_DB *idb)
{
	IDB = idb;
}

void network::initialise()
{
	//you are hear, initialise network with layers
}

void network::print_network_info()
{
	std::cout << "\n\nPrinting network information\n";
	std::cout << "---------------------------------------------------------------------";
	std::cout << "\nConnectivity: ";
	for (int i=0; i<layer_connectivities.size(); i++)
		std::cout << io::layer_connectivity_to_string(layer_connectivities[i]) << "\t";
	std::cout << "\nActivation_Function: ";
	for (int i=0; i<activation_functions.size(); i++)
		std::cout << io::activation_function_to_string(activation_functions[i]) << "\t";
	std::cout << "\nPool layer: ";
	for (int i=0; i<pools.size(); i++)
			std::cout << pools[i] << "\t";
	std::cout << "\nlayer width: " << field_width << std::endl;
	std::cout << "layer height: " << field_height << std::endl;
	std::cout << "stride x: " << stride_x << std::endl;
	std::cout << "stride y: " << stride_y << std::endl;
	std::cout << "zero pad x: " << zero_pad_x << std::endl;
	std::cout << "zero pad y: " << zero_pad_y << std::endl;
	std::cout << "learning_rate: " << learning_rate << std::endl;
}
