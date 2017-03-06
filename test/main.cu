/***************************************************************************//**
 * \file main.cu
 * \author Christopher Minar (minarc@oregonstate.edu)
 * \brief testing file
 */

#include "operators.h" //contains all device operator tests
#include "back_prop.h" //contains all back propogation kernel tests



int main()
{
	cudaDeviceReset();
	test_operators();
	test_back_prop();


	return 0;
}

//todo add a verbose mode to the back_prop indices tests
//the center pixel index is correct for propogate convolutional, work on the weights index
//then work on fully connected
//then actually put some numbers through these things

