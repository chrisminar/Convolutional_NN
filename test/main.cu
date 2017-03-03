/***************************************************************************//**
 * \file main.cu
 * \author Christopher Minar (minarc@oregonstate.edu)
 * \brief testing file
 */

#include <run.h>


#include "operators.h"



int main()
{
	cudaDeviceReset();
	test_operators();

	return 0;
}



