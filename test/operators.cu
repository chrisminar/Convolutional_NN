/***************************************************************************//**
 * \file operators.cu
 * \author Christopher Minar (minarc@oregonstate.edu)
 * \brief tests device operators
 */

#include <train.h>
#include "operators.h"

#include <random>
#include <vector>
#include <algorithm> //std::any_of

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

//tests all the operator functions
void test_operators()
{
	std::vector<bool> flags(4);
	flags[0] = test_square_op();
	flags[1] = test_dsig_op();
	flags[2] = test_dsig_from_sig_op();
	flags[3] = test_sig_mult_op();
	if ( std::any_of(flags.begin(), flags.end(), [](bool i){return i==false;} ) )
		std::cout<<"Not all operators passed!\n";
	else
		std::cout<<"All operators passed!\n";
}

//tests the square operator from train.h
bool test_square_op()
{
	thrust::device_vector<int> 	V1(10),
								V2(10);

	kernels::square<double> op;
	thrust::sequence(V1.begin(),V1.end());
	thrust::transform(V1.begin(),V1.end(), V2.begin(), op);
	bool flag = true;
	for (int i=0; i<V1.size(); i++)
	{
		if (V1[i]*V1[i] != V2[i])
			flag = false;
		//std::cout<<V1[i]<<" "<<V2[i]<<std::endl;
	}
	if (!flag)
		std::cout<<"Square unary operator failed!\n";
	else
		std::cout<<"Square unary operator passed!\n";

	return flag;
}

//tests the dsig operator from train.h
bool test_dsig_op()
{
	thrust::device_vector<double> 	V1(10),
									V2(10);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1,1);
	for (int i=0; i<V1.size(); i++)
		V1[i] = dis(gen);
	kernels::d_sig<double> op;
	thrust::transform(V1.begin(), V1.end(), V2.begin(), op);
	bool flag = true;
	for (int i=0; i<V1.size(); i++)
	{
		double sig = 1 / (1 + exp(-V1[i]));
		if (sig*(1-sig) != V2[i])
			flag = false;
	}
	if (!flag)
		std::cout<<"dsig unary operator failed!\n";
	else
		std::cout<<"dsig unary operator passed!\n";

	return flag;
}

//tests the dsig from sig operator from train.h
bool test_dsig_from_sig_op()
{
	thrust::device_vector<double> 	V1(10),
									V2(10),
									V3(10);

	//seed V1 with a normal distribution and make v2 the sigmoid of it
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1,1);
	for (int i=0; i<V1.size(); i++)
	{
		V1[i] = dis(gen);
		V2[i] = 1 / (1 + exp(-V1[i]));
	}

	//perform dsig from sig
	kernels::d_sig_from_sig<double> op;
	thrust::transform(V2.begin(), V2.end(), V3.begin(), op);

	//test
	bool flag = true;
	for (int i=0; i<V1.size(); i++)
	{
		if (abs(V2[i]*(1-V2[i]) - V3[i]) > 1e-10)
			flag = false;
	}
	if (!flag)
		std::cout<<"dsig from sig unary operator failed!\n";
	else
		std::cout<<"dsig from sig unary operator passed!\n";

	return flag;
}

//test the sig mult operator from train.h
bool test_sig_mult_op()
{
	thrust::device_vector<double> 	V1(10), //some sigmoid
									V2(10), //some constant
									V3(10); //operator

	//seed V1 with a normal distribution and make v2 the sigmoid of it
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-1,1);
	for (int i=0; i<V1.size(); i++)
	{
		V1[i] = 1 / (1 + exp(-dis(gen)));
		V2[i] = 3;
	}

	//perform sig mult
	kernels::sig_mult op;
	thrust::transform(V2.begin(), V2.end(), V1.begin(), V3.begin(), op);

	//test
	bool flag = true;
	for (int i=0; i<V1.size(); i++)
	{
		if (abs(V1[i]*(1-V1[i])*V2[i] - V3[i]) > 1e-10)
			flag = false;
	}
	if (!flag)
		std::cout<<"sig mult operator failed!\n";
	else
		std::cout<<"sig mult operator passed!\n";

	return flag;
}
