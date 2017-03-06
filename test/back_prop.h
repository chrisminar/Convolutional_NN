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
void print_dweight_fc_test(int ldo, int ld, int fs, int bs, std::string s, thrust::device_vector<int> w);
template <typename T>
void print_temp(int fw, int fh, int ldo, int batch_size, std::string s, thrust::device_vector<T> temp);
void test_back_prop();
bool test_calculate_dweight();
bool test_calculate_dweight_fc();
bool test_propogate_conv();

