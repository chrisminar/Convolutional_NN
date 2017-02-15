/***************************************************************************//**
 * \file types.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief This file stores all the enumerators
 */

#pragma once

enum activation_function
{
	SIGMOID,
	TANH,
	RELU
};

enum layer_connectivity
{
	CONVOLUTIONAL,
	FULLY_CONNECTED
};

enum layer_type
{
	INPUT,
	OUTPUT,
	HIDDEN
};
