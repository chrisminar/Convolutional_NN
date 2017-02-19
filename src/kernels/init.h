#pragma once
#include <curand.h>
#include <curand_kernel.h>

namespace kernels
{
__global__
void init_weights(double *weights, int filter_size, int layer_depth, int layer_depth_out, curandStateXORWOW *state);
__global__
void setup_rand(curandStateXORWOW *state, unsigned long seed);
}
