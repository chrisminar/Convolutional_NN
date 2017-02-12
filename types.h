/***************************************************************************//**
 * \file types.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief This file stores all the enumerators
 */

enum activation_function
{
	SIGMOID,
	TANH,
	RELU
};

enum layer_type
{
	CONVOLUTIONAL,
	FULLY_CONNECTED
};

enum layer_position
{
	INPUT,
	OUTPUT,
	HIDDEN
};
