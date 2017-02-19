#include "init.h"

namespace kernels
{
__global__
void init_weights(double *weights, int filter_size, int layer_depth, int layer_depth_out, curandStateXORWOW *state)
{
	//todo need to figure out bias terms still
	int number_of_values_per_weight = filter_size*filter_size*layer_depth;
	if (threadIdx.x + blockDim.x + blockIdx.x >= number_of_values_per_weight * layer_depth_out)
		return;
	int i = threadIdx.x + blockDim.x + blockIdx.x;
	weights[i] = curand_normal_double(state);
}

__global__
void setup_rand(curandStateXORWOW *state, unsigned long seed)
{
	int id = threadIdx.x + blockDim.x + blockIdx.x;
	curand_init (seed, id, 0, &state[id]);
}

}
