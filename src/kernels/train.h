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

__global__
void delta_FC(double *delta, double *weights, double *bias, int field_width, int field_height,
				int layer_depth, int field_width_out, int field_height_out, int layer_depth_out, int batch_size);

__global__
void delta_pb(double *delta, double *weights, double *bias, double *bias_delta, int field_width, int field_height,
				int layer_depth, int field_width_out, int field_height_out, int layer_depth_out, int batch_size);

__global__
void weight_delta(double *dweight, double *temp, double *dtemp, int filter_size, int field_height, int field_width, int layer_depth, int layer_depth_out, int batch_size);
}
