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

std::string layer_position_to_string(layer_type typ)
{
	if (typ == INPUT)
		return "INPUT";
	else if (typ == OUTPUT)
		return "OUTPUT";
	else if (typ == HIDDEN)
		return "HIDDEN";
	else
		return "NONE";
}

std::string layer_connectivity_to_string(layer_connectivity conv)
{
	if (conv == CONVOLUTIONAL)
		return "CONVOLUTIONAL";
	else if (conv == FULLY_CONNECTED)
		return "FULLY_CONNECTED";
	else
		return "NONE";
}

std::string activation_function_to_string(activation_function fn)
{
	if (fn == SIGMOID)
		return "SIGMOID";
	else if (fn == TANH)
		return "TANH";
	else if (fn == RELU)
		return "RELU";
	else
		return "NONE";
}
