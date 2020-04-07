#include "hdr.h"
#include <fstream>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace cv;
using namespace std;
#define ln_value 2.718281828

HDR::HDR(){}

void HDR::MTB() {
    Mat sample = inputArrays[0].clone();
    vector<int> move_step_x(inputArrays.size(), 0);
    vector<int> move_step_y(inputArrays.size(), 0);

    for(int j = 1; j < inputArrays.size(); j++) {
        if(j == 0){
            turnToGray(sample, sample);
        } else {
            Mat temp1 = inputArrays[j].clone();
            turnToGray(temp1, temp1);
            for(int i = 5; i >= 0; i--) {
                Mat temp_samp = sample.clone();
                Mat temp = temp1.clone();

                Mat sample_img = sample.clone();
                Mat other_img = temp1.clone();

                Mat samp = temp_samp.clone();
                Mat other = temp1.clone();
                resize(samp,samp,Size((int)(sample.cols / pow(2, i+1)), (int)(sample.rows / pow(2, i+1))));
                resize(samp,samp,Size((int)(sample.cols / pow(2, i)), (int)(sample.rows / pow(2, i))));
                resize(other,other,Size((int)(sample.cols / pow(2, i+1)), (int)(sample.rows / pow(2, i+1))));
                resize(other,other,Size((int)(sample.cols / pow(2, i)), (int)(sample.rows / pow(2, i))));

                for(int b = 0; b < samp.rows; b++) {
                    for(int a = 0; a < samp.cols; a++) {
                        if(samp.at<uchar>(b,a) != sample_img.at<uchar>(b,a)) {
                            temp_samp.at<uchar>(b,a) = 0;
                        }
                        if(other.at<uchar>(b,a) != other_img.at<uchar>(b,a)) {
                            temp.at<uchar>(b,a) = 0;
                        }
                        if(sample_img.at<uchar>(b,a) != other_img.at<uchar>(b,a)) {
                            temp.at<uchar>(b,a) = 0;
                        }
                    }
                }

                double total[9] = {0};

                for(int b = 0; b < temp_samp.rows; b++) {
                    for(int a = 0; a < temp_samp.cols; a++) {
                        int count = 0;
                        for(int y = -1; y <= 1; y++) {
                            for(int x = -1; x <= 1; x++) {
                                if(a + move_step_x[j] + x > 0 && a + move_step_x[j] + x < temp.cols && b + move_step_y[j] + y > 0 && b + move_step_y[j] + y < temp.rows) {
                                    if((int)temp_samp.at<uchar>(b, a) != (int)temp.at<uchar>(b + move_step_y[j] + y, a + move_step_x[j] + x))
                                        total[count]++;
                                }
                                count++;
                            }
                        }
                    }
                }

                int a = 0;
                for(int b = 1; b < 9; b++) {
                    if(total[a] > total[b]) {
                        a = b;
                    }
                }

                if(total[a] * 1.5 >= total[4]) {
                    a = 4;
                }

                switch(a) {
                    case 0:
                        move_step_y[j] = (move_step_y[j] - 1) * 2;
                        move_step_x[j] = (move_step_x[j] - 1) * 2;
                        break;
                    case 1:
                        move_step_y[j] = (move_step_y[j] - 1) * 2;
                        break;
                    case 2:
                        move_step_y[j] = (move_step_y[j] - 1) * 2;
                        move_step_x[j] = (move_step_x[j] + 1) * 2;
                        break;
                    case 3:
                        move_step_x[j] = (move_step_x[j] - 1) * 2;
                        break;
                    case 4:
                        break;
                    case 5:
                        move_step_x[j] = (move_step_x[j] + 1) * 2;
                        break;
                    case 6:
                        move_step_y[j] = (move_step_y[j] + 1) * 2;
                        move_step_x[j] = (move_step_x[j] - 1) * 2;
                        break;
                    case 7:
                        move_step_y[j] = (move_step_y[j] + 1) * 2;
                        break;
                    case 8:
                        move_step_y[j] = (move_step_y[j] + 1) * 2;
                        move_step_x[j] = (move_step_x[j] + 1) * 2;
                        break;
                }
            }
        }
    }

    int minX = move_step_x[0], maxX = move_step_x[0];
    int minY = move_step_y[0], maxY = move_step_y[0];

    for(int i = 1; i < inputArrays.size(); i++) {
        if(minX > move_step_x[i]){minX = move_step_x[i];}
        if(maxX < move_step_x[i]){maxX = move_step_x[i];}
        if(minY > move_step_y[i]){minY = move_step_y[i];}
        if(maxY < move_step_y[i]){maxY = move_step_y[i];}
    }
    vector<Mat> tempdest(inputArrays.size());

    int a, j, i, k;
    #pragma parallel for private(a, j, i, k)
    for(a=0; a < inputArrays.size(); a++) {
        Mat&& dest = Mat::zeros(sample.rows + abs(minY) + abs(maxY), sample.cols + abs(minX) + abs(maxX), CV_8UC3);
        for(j = 0; j < inputArrays[a].rows; j++) {
            for(i = 0; i < inputArrays[a].cols; i++) {
                for(k = 0; k < 3; k++) {
                        dest.at<Vec3b>(j + abs(minY) + move_step_y[a], i + abs(minX) + move_step_x[a])[k] = inputArrays[a].at<Vec3b>(j, i)[k];
                }
            }
        }
        tempdest[a] = dest.clone();
    }

    Mat&& dest1 = Mat::zeros(sample.rows + abs(minY) + abs(maxY), sample.cols + abs(minX) + abs(maxX), CV_8UC3);
    int tmp = 0;
    for(j = 0; j < tempdest[0].rows; j++) {
        for(i = 0; i < tempdest[0].cols; i++) {
            for(k = 0; k < 3; k++) {
                tmp = 0;
                for(a=0; a < tempdest.size(); a++)
                {
                     tmp += tempdest[a].at<Vec3b>(j, i)[k];
                }

                dest1.at<Vec3b>(j, i)[k] = tmp / tempdest.size();
            }
        }
    }
    inputArrays = tempdest;
}

void HDR::turnToGray(Mat &inputArray, Mat &outputArray) {
    int color_count[256] = {0};
    int i, j;
    Mat temp;
    // 轉為黑白
    cvtColor(inputArray, temp, COLOR_BGR2GRAY);

    for(j = 0; j < inputArray.rows; j++) {
        for(i = 0; i < inputArray.cols; i++) {
            color_count[temp.at<uchar>(j, i)]++;
        }
    }

    int threshold_total = 0;
    double thresh;
    for(j = 0; j < inputArray.rows; j++) {
        threshold_total += color_count[j];
        if(threshold_total >= (temp.cols * temp.rows)/2) {
            thresh = j;
            break;
        }
    }

    Mat&& dest = Mat::zeros(temp.rows, temp.cols, CV_8UC1);

    for(j = 0; j < inputArray.rows; j++) {
        for(i = 0; i < inputArray.cols; i++) {
            temp.at<uchar>(j, i) > thresh ? dest.at<uchar>(j, i) = 255 : dest.at<uchar>(j, i) = 0;
        }
    }
    outputArray.release();
    inputArray = dest.clone();
}

void HDR::getInputImage(string testing_set) {
    cout << "getInputImage" << endl;
    for(int i = 0; i < file_name_list.size();i++){
        Mat temp = imread("../test_image/" + testing_set + "/" + file_name_list[i], IMREAD_COLOR);
        inputArrays.push_back(temp);
    }
    cout << "圖片張數：" << inputArrays.size() << endl;
}

void HDR::getExposureTime(string testing_set) {
    cout << "getExposureTime" << endl;

    fstream file;
    string name;
    double time;
    file.open("../test_image/" + testing_set + "/setting.txt",ios::in);
    if(!file) {    //檢查檔案是否成功開啟
        cerr << "Can't open file!\n";
        exit(1);     //在不正常情形下，中斷程式的執行
    }

    while(file >> name >> time) {
        // cout << setw(4) << setiosflags(ios::right) << name << setw(8) << setiosflags(ios::right) << time << endl;
        file_name_list.push_back(name);
        exposure_time_list.push_back(time);
    }

}

void HDR::getRadianceMap() {
    cout << "getRadianceMap" << endl;

    int re_w = 10, re_h = 10;
    vector<vector<int>> RZ;
    vector<vector<int>> GZ;
    vector<vector<int>> BZ;

    for(int i = 0; i < re_h; i++) {
        for(int j = 0; j < re_w; j++) {
            vector<int> value_r;
            vector<int> value_g;
            vector<int> value_b;
            for(int a = 0; a < inputArrays.size(); a++) {
                Mat temp;
                resize(inputArrays[a], temp, Size(inputArrays[a].cols, inputArrays[a].rows));
                resize(temp, temp, Size(re_w, re_h));
                value_r.push_back(temp.at<Vec3b>(i, j)[2]);
                value_g.push_back(temp.at<Vec3b>(i, j)[1]);
                value_b.push_back(temp.at<Vec3b>(i, j)[0]);
            }
            RZ.push_back(value_r);
            GZ.push_back(value_g);
            BZ.push_back(value_b);
        }
    }

    vector<Mat> MZ(3);
    MZ[2] = lnE(RZ, exposure_time_list).clone();
    MZ[1] = lnE(GZ, exposure_time_list).clone();
    MZ[0] = lnE(BZ, exposure_time_list).clone();

    Mat hdr_img(inputArrays[0].rows, inputArrays[0].cols, CV_32FC3);

    for(int color_length = 0; color_length < 3; color_length++) {
        for(int i = 0; i < hdr_img.rows; i++) {
            for(int j = 0; j < hdr_img.cols; j++) {
                double weight_sum = 0;
                double total = 0;
                for(int input_length = 0; input_length < inputArrays.size(); input_length++) {
                    int color = (int)inputArrays[input_length].at<Vec3b>(i, j)[color_length];
                    double w = Weighting(color + 1);
                    total += (double)(w * (MZ[color_length].at<double>(color, 0) - (log(exposure_time_list[input_length])/log(ln_value))));
                    weight_sum += w;
                }
                if(!IsFiniteNumber((float)exp(total / weight_sum))) {
                    hdr_img.at<Vec3f>(i, j)[color_length] = 0.0;
                } else {
                    hdr_img.at<Vec3f>(i, j)[color_length] = (float)exp(total / weight_sum);
                }
            }
        }
    }

    outputArray.release();
    outputArray = hdr_img.clone();
}

Mat HDR::lnE(vector<vector<int> > Z,vector<double> exposure_time_list) {
    vector<vector<double>> A_matrix;
    vector<double> b_matrix;
    int n = 256;
    double lambda = 10;
    for (int i = 0; i < Z.size() * Z.at(0).size() + n + 1; i++) {
        vector< double > temp(n + Z.size(), 0);
        A_matrix.push_back(temp);
    }
    b_matrix.resize(A_matrix.size());

    int k = 0;
    for (int i = 0; i < Z.size(); i++) {
        for (int j = 0; j < Z.at(0).size(); j++) {
            double wij = Weighting(Z[i][j] + 1);
            b_matrix[k] = wij * log(exposure_time_list[j]) / log(ln_value);
            A_matrix[k][Z[i][j]] = wij;
            A_matrix[k][n + i] = -wij;
            k++;
        }
        if(i == Z.size() - 1){
            A_matrix[k][128] = 1;
            k++;
        }
    }

    for (int i = 0; i < n-1; i++) {
        double temp = Weighting(i + 2);
        A_matrix[k][i] = lambda * temp;
        A_matrix[k][i + 1] = -2 * lambda * temp;
        A_matrix[k][i + 2] = lambda * temp;
        k++;
    }

    Mat AA_matrix(A_matrix.size(), A_matrix[0].size(),CV_64F);
    Mat bb_matrix(b_matrix,true);

    for (int i = 0; i < AA_matrix.rows; i++) {
        bb_matrix.at<double>(i) = b_matrix[i];
        for (int j = 0; j < AA_matrix.cols; j++) {
            AA_matrix.at<double>(i, j) = A_matrix[i][j];
        }
    }

    Mat ans(AA_matrix.rows, 1,CV_64F);
    solve(AA_matrix, bb_matrix, ans, DECOMP_SVD);

    return ans;
}

double HDR::Weighting(int z) {
    int max = 255, min = 0;
    if (z <= 0.5 * (min + max + 1)) {
        return (z - min) / 128.0;
    } else {
        return (max + 1 - z) / 128.0;
    }
}

void HDR::toneMapping() {
    cout << "toneMapping" << endl;
    Mat intensity(outputArray.rows, outputArray.cols, CV_32FC1);
    Mat temp(outputArray.rows, outputArray.cols, CV_32FC1);
    Mat result(outputArray.rows, outputArray.cols, CV_8UC3);
    double Mean_total = 0;
    for(int i = 0;i < outputArray.rows;i++){
        for(int j = 0;j < outputArray.cols;j++){
            float temp_intensity = 0.2126 * outputArray.at<Vec3f>(i, j)[2] + 0.7152 * outputArray.at<Vec3f>(i, j)[1] + 0.0722 * outputArray.at<Vec3f>(i, j)[0];
            Mean_total += log(0.00001 + temp_intensity);
        }
    }
    float Mean = exp(Mean_total / (double)(outputArray.rows * outputArray.cols));

    float a = 0.36; //參考Paper內數值 0.18 ~ 0.36
    float white = 3.0; //參考Paper

    for(int i = 0;i < outputArray.rows;i++){
        for(int j = 0;j < outputArray.cols;j++){
            float temp_intensity = 0.2126 * outputArray.at<Vec3f>(i, j)[2] + 0.7152 * outputArray.at<Vec3f>(i, j)[1] + 0.0722 * outputArray.at<Vec3f>(i, j)[0];
            float ld = a * temp_intensity / Mean;
            if(ld >= white) {
                temp.at<float>(i, j) = 1;
            } else {
                float c = 1 + (ld / (white * white));
                temp.at<float>(i, j) = ld * c / (1.0 + ld);
            }
            result.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(outputArray.at<Vec3f>(i, j)[0] * (temp.at<float>(i, j) * 255.0 / temp_intensity));
            result.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(outputArray.at<Vec3f>(i, j)[1] * (temp.at<float>(i, j) * 255.0 / temp_intensity));
            result.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(outputArray.at<Vec3f>(i, j)[2] * (temp.at<float>(i, j) * 255.0 / temp_intensity));
        }
    }

    outputArray.release();
    outputArray = result.clone();
}


void HDR::writeHdrImage() {
    cout << "writeHdrImage" << endl;
    imwrite("../result/final.hdr",outputArray);
}

void HDR::writeJpgImage() {
    cout << "writeJpgImage" << endl;
    imwrite("../result/final.jpg",outputArray);
}
