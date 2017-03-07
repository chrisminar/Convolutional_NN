/***************************************************************************//**
 * \file train_test.cu
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief implementation of the training test kernels
 */

#include "train_test.h"
#include <stdio.h>

namespace kernels
{

__global__
void calculate_dweight_test(double *dweight, double *input, double *ddot, int *pool_flag,
							int filter_size, int field_height, int field_width,
							int layer_depth, int layer_depth_out, int batch_size,
							int *wi_t, int *oli_t, int *oln_t, int *ili_t,
							int *iln_t, int *ly_t, int *lx_t,
							int *count_t, double *sum_t, int *input_index_t, int *ddot_index_t)
{
	int weight_index = threadIdx.x + blockDim.x * blockIdx.x,						//weight array index
		outer_layer_index = weight_index % (filter_size*filter_size*layer_depth),	//count from 0 to fs*fs*ld for each layer out
		outer_layer_number = weight_index / (filter_size*filter_size*layer_depth),	//outer layer this weight is in
		inner_layer_index = outer_layer_index % (filter_size*filter_size),			//count from 0 to fs*fs for each inner layer
		inner_layer_number = outer_layer_index / (filter_size*filter_size),			//inner layer this weight is in
		layer_y = inner_layer_index/filter_size,									//y position in layer
		layer_x = inner_layer_index%filter_size;									//x position in layer

	wi_t[weight_index] = weight_index;
	oli_t[weight_index] = outer_layer_index;
	oln_t[weight_index] = outer_layer_number;
	ili_t[weight_index] = inner_layer_index;
	iln_t[weight_index] = inner_layer_number;
	ly_t[weight_index] = layer_y;
	lx_t[weight_index] = layer_x;

	if (weight_index >= filter_size*filter_size*layer_depth*layer_depth_out)
		return;

	double sum=0;
	int count=0;
	int input_index;
	int ddot_index;
	int flag=1;
	int loop_index = 0;

	//step through images
	for (int m=0; m<batch_size; m++)
	{
		//step through field y
		for (int j=0; j<field_height; j++)
		{
			//step through field x
			for (int i=0; i<field_width; i++)
			{
				input_index =	m * field_width*field_height*layer_depth +
								inner_layer_number * field_width*field_height +
								j * field_width +
								i;
				ddot_index = 	m * field_width*field_height*layer_depth_out +
								outer_layer_number * field_width*field_height +
								j * field_width +
								i;
				loop_index = m * field_height*field_width +
								j*field_width+
								i;
				input_index_t[weight_index * batch_size*field_height*field_width + loop_index] = input_index;
				ddot_index_t[weight_index * batch_size*field_height*field_width + loop_index] = ddot_index;
				//if were out of bounds, mult this weight*temp by zero
				if ( (layer_y == 0 and j==field_height-1) || (layer_y==filter_size-1 and j==0) || (layer_x == 0 and i==field_width-1) || (layer_x==filter_size-1 and i == 0) ) //todo might be able to cut out this if statement by modifiying the above for loops
					flag = 0;
				else
					flag = 1;
				sum += input[input_index] * ddot[ddot_index] * flag;// * pool_flag[i];
				count +=1*flag;
			}
		}
	}
	dweight[weight_index] = sum;
	sum_t[weight_index] = sum;
	count_t[weight_index] = count;
}

//can't handle things that are not 1x1xsomething
__global__
void calculate_fc_dweight_test(double *dweight, double *input, double *ddot,
								int filter_size, int field_height, int field_width,
								int layer_depth, int layer_depth_out, int batch_size,
								int *wi_t, int *oli_t, int *oln_t, int *ili_t,
								int *iln_t, int *ly_t, int *lx_t,
								int *input_index_t, int *ddot_index_t)
{
	//some useful numbers
	int pixel_per_ol = filter_size*filter_size*layer_depth,
		pixel_per_il = filter_size*filter_size;

	//indices
	int weight_index = threadIdx.x + blockDim.x * blockIdx.x;						//weight array index
	int	outer_layer_index = weight_index % (pixel_per_ol);							//count from 0 to fs*fs*ld for each layer out
	int	outer_layer_number = weight_index / (pixel_per_ol);							//outer layer this weight is in
	int	inner_layer_index = outer_layer_index % (pixel_per_il);						//count from 0 to fs*fs for each inner layer
	int	inner_layer_number = outer_layer_index / (pixel_per_il);					//inner layer this weight is in
	int	layer_y = inner_layer_index/filter_size;									//y position in layer
	int	layer_x = inner_layer_index%filter_size;									//x position in layer

	if (weight_index >= filter_size*filter_size*layer_depth*layer_depth_out)
		return;
	wi_t[weight_index] = weight_index;
	oli_t[weight_index] = weight_index % pixel_per_ol;
	oln_t[weight_index] = outer_layer_number;
	ili_t[weight_index] = inner_layer_index;
	iln_t[weight_index] = inner_layer_number;
	ly_t[weight_index] = layer_y;
	lx_t[weight_index] = layer_x;
	double sum=0;
	int input_index;
	int ddot_index;

	//step through images
	for (int m=0; m<batch_size; m++)
	{
		ddot_index =	1*1*layer_depth_out*m + //1x1 should be be field width out and field height out
						1*1*outer_layer_number;
		input_index =	m * field_width*field_height*layer_depth +
						inner_layer_number * field_width*field_height +
						layer_y * field_width +
						layer_x;
		input_index_t[weight_index * batch_size + m] = input_index;
		ddot_index_t[weight_index * batch_size + m] = ddot_index;
		sum += input[input_index] * ddot[ddot_index];
	}
	dweight[weight_index] = sum;
}
/*
__global__
void propogate_ddot_fc_test(double *ddot, double *ddot_upstream, double *weights, double *bias,
						int field_height, int field_width, int layer_depth_out, int filter_size,
						int field_height_us, int field_width_us, int layer_depth_out_us, int batch_size)
{
	//some useful numbers
	int num_pixels_per_ddot_layer = 1,
		num_pixels_per_ddot_image = num_pixels_per_ddot_layer*layer_depth_out,
		num_pixels_per_layer_us = field_width_us*field_height_us,
		num_pixels_per_image_us = num_pixels_per_layer_us*layer_depth_out_us;

	//figure out where this kernel is
	int output_index = threadIdx.x + blockDim.x * blockIdx.x,		// ddot index
		image_number = output_index / (num_pixels_per_image_us),	// image this ddot is in
		image_index = output_index % (num_pixels_per_image_us),		// count from 0 to num_pixels_per_image in this image
		layer_us_number = image_index / (num_pixels_per_layer_us),	// output layer this image is in
		layer_us_index = image_index % (num_pixels_per_layer_us),	// count from 0 to num_pixels_per_layer in this output layer
		field_x = layer_us_index % field_width_us,					// ddot x position in output layer
		field_y = layer_us_index / field_width_us;					// ddot y position in output layer

	//if were outside of the minibatch range return
	if (output_index >= num_pixels_per_image_us * batch_size)
		return;

	//setup dot product
	double sum = 0;
	int ddot_index = 0;
	int weight_index = 0;

	//loop over layer depth
	for (int k=0; k<layer_depth_out; k++)
	{
		ddot_index =	image_number*num_pixels_per_ddot_image +		// past images
						k*num_pixels_per_ddot_layer;					// layers
		weight_index = 	k*field_width*field_height*layer_depth_out +	// layer outs
						layer_us_number*field_height*field_width +		// layer in
						field_y*(field_height/field_height_us)*field_width +	// filter rows
						field_x*(field_width/field_width_us);			// filter columns note: each filter is rotated by 180, which is why we go the the max of the filter layer then subtract off
		sum += ddot[ddot_index] * weights[weight_index];
	}//endk
	ddot_upstream[output_index] = sum; //todo deal with bias
}
*/
//note: not setup for alternate filter sizes, zero padding or strides
//note: not sure if we need a pool flag for this kernel
__global__
void propogate_ddot_conv_test(double *ddot, double *ddot_upstream, double *weights,
								int field_height, int field_width, int layer_depth_out, int filter_size,
								int field_height_us, int field_width_us, int layer_depth_out_us, int batch_size,
								int *oi_t, int *in_t, int *ii_t, int *lun_t, int *lui_t, int *fx_t, int *fy_t,
								double *cpi_t, double *wi_t)
{
	//some useful numbers
	int num_pixels_per_layer = field_width*field_height,
		num_pixels_per_image = num_pixels_per_layer*layer_depth_out,
		num_pixels_per_layer_us = field_width_us*field_height_us,
		num_pixels_per_image_us = num_pixels_per_layer_us*layer_depth_out_us;

	//figure out where this kernel is
	int output_index = threadIdx.x + blockDim.x * blockIdx.x,		// ddot index
		image_number = output_index / (num_pixels_per_image_us),	// image this ddot is in
		image_index = output_index % (num_pixels_per_image_us),		// count from 0 to num_pixels_per_image in this image
		layer_us_number = image_index / (num_pixels_per_layer_us),	// output layer this image is in
		layer_us_index = image_index % (num_pixels_per_layer_us),	// count from 0 to num_pixels_per_layer in this output layer
		field_x = layer_us_index % field_width_us,					// ddot x position in output layer
		field_y = layer_us_index / field_width_us;					// ddot y position in output layer

	//if were outside of the minibatch range return
	if (output_index >= num_pixels_per_image_us * batch_size)
		return;

	oi_t[output_index] = output_index;
	in_t[output_index] = image_number;
	ii_t[output_index] = image_index;
	lun_t[output_index] = layer_us_number;
	lui_t[output_index] = layer_us_index;
	fy_t[output_index] = field_y;
	fx_t[output_index] = field_x;

	//setup dot product
	double sum = 0;
	int center_pixel_index = 0;
	int weight_index = 0;
	//the weight is going to be filter_size * filter_size * layer depth_in. There is one weight for each output layer
	//loop over output depth
	int filter_half = filter_size/2;
	//loop over layer depth
	for (int k=0; k<layer_depth_out; k++)
	{
		center_pixel_index =	image_number*num_pixels_per_image + 				// past images
								k*num_pixels_per_layer +							// layers
								field_y*field_height/field_height_us*field_width +	// rows
								field_x*field_width/field_width_us;					// columns
		cpi_t[output_index*layer_depth_out + k] = center_pixel_index;
		//loop over filter_y
		for (int j=-filter_half; j<=filter_half; j++)
		{
			//loop over filter_x
			for (int i=-filter_half; i<=filter_half; i++)
			{
				//check if we are at a boundary to account for zero padding
				//     						 left                         				  right
				if ( (field_x*field_width/field_width_us + i < 0) || (field_x*field_width/field_width_us + i >= field_width) ||
						//							bottom                   								  top
						(field_y*field_height/field_height_us*field_width + j < 0) || (field_y*field_height/field_height_us + j >= field_height) )
				{}
				else
				{
					weight_index = 			k*layer_depth_out_us*filter_size*filter_size +				// layer outs
											layer_us_number*filter_size*filter_size +				// layer in
											filter_size*filter_size - 1 -							// maximum filter layer
											filter_size*(j+filter_half) -							// filter rows
											(i+filter_half);										// filter columns note: each filter is rotated by 180, which is why we go the the max of the filter layer then subtract off
					sum += ddot[center_pixel_index + j*field_width + i] * weights[weight_index];	//todo multiplier for center pixel index is probably wrong
					wi_t[output_index*layer_depth_out*filter_size*filter_size +
						 k*filter_size*filter_size +
						 (j+filter_half)*filter_size +
						 (i+filter_half)] = weight_index;
				}//endif
			}//endi
		}//endj
	}//endk
	ddot_upstream[output_index] = sum; //todo deal with bias
}

}
