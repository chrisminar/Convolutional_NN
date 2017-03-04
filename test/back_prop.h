/***************************************************************************//**
 * \file back_prop.h
 * \author Christopher Minar (minarc@oregonstate.edu)
 * \brief declaration of back_propogation kernel unit tests
 */

#pragma once
#include <string>
#include <thrust/device_vector.h>

template<typename T>
void print_weights(int ldo, int ld, int fs, std::string s, thrust::device_vector<T> w);
void print_dweight_test_loop(int ldo, int ld, int fs, int bs, int fw, int fh, std::string s, thrust::device_vector<int> w);
void test_back_prop();
bool test_calculate_dweight();
bool test_calculate_dweight_fc();

