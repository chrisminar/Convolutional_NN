/***************************************************************************//**
 * \file main.cu
 * \author Christopher Minar (minarc@oregonstate.edu)
 * \brief Main source-file of \c convNet
 */

#include "io.h"
#include "types.h"
#include "image_DB.h"

int main()
{
	image_DB idb;
    io::read_CIFAR10(idb);
    return 0;
}
