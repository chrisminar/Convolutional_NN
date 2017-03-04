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
	test_back_prop();//look at the last to indices of test the fc test, then test propogate kernels


	return 0;
}



