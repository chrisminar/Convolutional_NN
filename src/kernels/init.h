#pragma once
#include <curand.h>
#include <curand_kernel.h>

namespace kernels
{
__global__
void init_weights(double *weights, int filter_size, int layer_depth, int layer_depth_out, curandState_t state);
__global__
void setup_rand(curandState *state, unsigned long seed);
}
