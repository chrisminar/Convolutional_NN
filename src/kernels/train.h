/***************************************************************************//**
 * \file train.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of training kernels
 */

#pragma once

namespace kernels
{
//unary operator that returns the squared input
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const {
            return x * x;
        }
};

//unary operator that returns the derivative of the sigmoid function
template <typename T>
struct d_sig
{
    __host__ __device__
        T operator()(const T& x) const
    	{
    		double sig =  1 / (1 + exp(x));
            return sig*(1-sig);
        }
};
}
