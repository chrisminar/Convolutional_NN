/***************************************************************************//**
 * \file train.h
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Declaration of training kernels
 */

#include <thrust/functional.h>
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
    		double sig =  1 / (1 + exp(-x));
            return sig*(1-sig);
        }
};

//unary operator that returns the derivative of the sigmoid function
template <typename T>
struct d_sig_from_sig
{
    __host__ __device__
        T operator()(const T& x) const
    	{
            return x*(1-x);
        }
};

/*
 * binary operator that takes the first argument and multiplies it by the derivative of a sigmoid
 * i.e.
 * input a,b
 * where a is some value and b = sigmoid(c)
 * output = a*(1-b)*b = a*sigmoid'(c)
 */
struct sig_mult : public thrust::binary_function<double, double, double>
{
	__host__ __device__
	double operator() (double a, double b) { return a*(1-b)*b; }
};
__global__
void calculate_dweight(double *dweight, double *input, double *ddot, int *pool_flag, int filter_size, int field_height, int field_width, int layer_depth, int layer_depth_out, int batch_size)
;__global__
void calculate_fc_dweight(double *dweight, double *temp, double *dtemp, int filter_size, int field_height, int field_width, int layer_depth, int layer_depth_out, int batch_size);
__global__
void propogate_ddot_conv(double *ddot, double *ddot_upstream, double *weights, double *bias,
								int field_height, int field_width, int layer_depth_out, int filter_size,
								int field_height_us, int field_width_us, int layer_depth_out_us, int batch_size);
__global__
void propogate_ddot_fc(double *ddot, double *ddot_upstream, double *weights, double *bias,
						int field_height, int field_width, int layer_depth_out, int filter_size,
						int field_height_us, int field_width_us, int layer_depth_out_us, int batch_size);
}
