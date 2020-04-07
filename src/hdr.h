#ifndef HDR_H
#define HDR_H

#include "opencv2/opencv.hpp"
#include <iomanip>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


class HDR {
  public:
    HDR();
    void getInputImage(string testing_set);
    void getExposureTime(string testing_set);
    void getRadianceMap();
    void toneMapping();
    void writeHdrImage();
    void writeJpgImage();
    Mat lnE(vector<vector<int> > Zij,vector<double> exposure_time_list);
    void turnToGray(Mat &inputArray, Mat &outputArray);
    void MTB();
    double Weighting(int  z);
  private:
    vector<Mat> inputArrays;
    Mat outputArray;
    vector<string> file_name_list;
    vector<double> exposure_time_list;

    bool IsFiniteNumber(double x) {
        return (x <= DBL_MAX && x >= -DBL_MAX);
    }
};

#endif
