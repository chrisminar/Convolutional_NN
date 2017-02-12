/***************************************************************************//**
 * \file io.cpp
 * \author Christopher Minar (chrisminar@gmail.com)
 * \brief Implementation of the namespace io
 * // readCIFAR10.cc
 * \
 * \feel free to use this code for ANY purpose
 * \author : Eric Yuan
 * \my blog: http://eric-yuan.me/
 */

#include <fstream>
#include <iostream>

using namespace std;

namespace io
{
void read_batch(string filename, thrust::host_vector &vec, thrust::host_vector &label)
{
	//open batch file
    ifstream file (filename, ios::binary);
    //if we sucessfully opened the file...
    if (file.is_open())
    {
    	//number of images in the batch file
        int number_of_images = 10000;
        //each image is 32x32x3
        int n_rows = 32;
        int n_cols = 32;
        //for every image...
        //for(int i = 0; i < number_of_images; ++i)
        //test for just the first image
        for (int i = 0; i<1; i++)
        {
        	//read the data label
            unsigned char cifar_10_label = 0;
            file.read((char*) &cifar_10_label, sizeof(cifar_10_label));
            //make an empty vector of 3 matrices
            //each matrix will store one of the three color values
            vector<Mat> channels;
            //finished image matrix
            Mat fin_img = Mat::zeros(n_rows, n_cols, CV_8UC3);
            for(int ch = 0; ch < 3; ++ch){
                Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
                for(int r = 0; r < n_rows; ++r){
                    for(int c = 0; c < n_cols; ++c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp.at<uchar>(r, c) = (int) temp;
                    }
                }
                channels.push_back(tp);
            }
            merge(channels, fin_img);
            vec.push_back(fin_img);
            label.ATD(0, i) = (double)tplabel;
        }
    }
}


}//end namespace io


void
read_CIFAR10(Mat &trainX, Mat &testX, Mat &trainY, Mat &testY){

    string filename;
    filename = "cifar-10-batches-bin/data_batch_1.bin";
    vector<Mat> batch1;
    Mat label1 = Mat::zeros(1, 10000, CV_64FC1);
    read_batch(filename, batch1, label1);

    filename = "cifar-10-batches-bin/data_batch_2.bin";
    vector<Mat> batch2;
    Mat label2 = Mat::zeros(1, 10000, CV_64FC1);
    read_batch(filename, batch2, label2);

    filename = "cifar-10-batches-bin/data_batch_3.bin";
    vector<Mat> batch3;
    Mat label3 = Mat::zeros(1, 10000, CV_64FC1);
    read_batch(filename, batch3, label3);

    filename = "cifar-10-batches-bin/data_batch_4.bin";
    vector<Mat> batch4;
    Mat label4 = Mat::zeros(1, 10000, CV_64FC1);
    read_batch(filename, batch4, label4);

    filename = "cifar-10-batches-bin/data_batch_5.bin";
    vector<Mat> batch5;
    Mat label5 = Mat::zeros(1, 10000, CV_64FC1);
    read_batch(filename, batch5, label5);

    filename = "cifar-10-batches-bin/test_batch.bin";
    vector<Mat> batcht;
    Mat labelt = Mat::zeros(1, 10000, CV_64FC1);
    read_batch(filename, batcht, labelt);

    Mat mt1 = concatenateMat(batch1);
    Mat mt2 = concatenateMat(batch2);
    Mat mt3 = concatenateMat(batch3);
    Mat mt4 = concatenateMat(batch4);
    Mat mt5 = concatenateMat(batch5);
    Mat mtt = concatenateMat(batcht);

    Rect roi = cv::Rect(mt1.cols * 0, 0, mt1.cols, trainX.rows);
    Mat subView = trainX(roi);
    mt1.copyTo(subView);
    roi = cv::Rect(label1.cols * 0, 0, label1.cols, 1);
    subView = trainY(roi);
    label1.copyTo(subView);

    roi = cv::Rect(mt1.cols * 1, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt2.copyTo(subView);
    roi = cv::Rect(label1.cols * 1, 0, label1.cols, 1);
    subView = trainY(roi);
    label2.copyTo(subView);

    roi = cv::Rect(mt1.cols * 2, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt3.copyTo(subView);
    roi = cv::Rect(label1.cols * 2, 0, label1.cols, 1);
    subView = trainY(roi);
    label3.copyTo(subView);

    roi = cv::Rect(mt1.cols * 3, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt4.copyTo(subView);
    roi = cv::Rect(label1.cols * 3, 0, label1.cols, 1);
    subView = trainY(roi);
    label4.copyTo(subView);

    roi = cv::Rect(mt1.cols * 4, 0, mt1.cols, trainX.rows);
    subView = trainX(roi);
    mt5.copyTo(subView);
    roi = cv::Rect(label1.cols * 4, 0, label1.cols, 1);
    subView = trainY(roi);
    label5.copyTo(subView);

    mtt.copyTo(testX);
    labelt.copyTo(testY);

}

int
main()
{

    Mat trainX, testX;
    Mat trainY, testY;
    trainX = Mat::zeros(1024, 50000, CV_64FC1);
    testX = Mat::zeros(1024, 10000, CV_64FC1);
    trainY = Mat::zeros(1, 50000, CV_64FC1);
    testX = Mat::zeros(1, 10000, CV_64FC1);

    read_CIFAR10(trainX, testX, trainY, testY);

    return 0;
}
