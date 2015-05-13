//
//  ColorTransfer.cpp
//  ColorTransfer
//
//  Created by Hana_Chang on 2015/5/13.
//  Copyright (c) 2015年 Andrea.C. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include "ColorTransfer.h"
#include <vector>

using cv::Mat;
using cv::Mat_;
using cv::Scalar;
using std::vector;

ColorTransfer::ColorTransfer() {
    m_src = NULL;
    m_target = NULL;
}

void ColorTransfer::type_convert2double(const Mat& in, Mat& out) {
    in.convertTo(out, CV_64FC3);
}

void ColorTransfer::type_convert2uchar(const Mat& in, Mat& out) {
    in.convertTo(out, CV_8UC1);
}

void ColorTransfer::rgb2vec(const Mat& rgb3n, Mat& rgb1n) {
    vector<Mat> channels(3);
    split(rgb3n, channels);
    
    channels[2].reshape(0, 1).copyTo(rgb1n.row(0));
    channels[1].reshape(0, 1).copyTo(rgb1n.row(1));
    channels[0].reshape(0, 1).copyTo(rgb1n.row(2));
}

void ColorTransfer::vec2rgb(const Mat& rgb1n, Mat& rgb3n) {
    vector<Mat> channels(3);
    channels[0] = rgb1n.row(2).reshape(0, m_src->rows);
    channels[1] = rgb1n.row(1).reshape(0, m_src->rows);
    channels[2] = rgb1n.row(0).reshape(0, m_src->rows);
    
    merge(channels, rgb3n);
}

void ColorTransfer::rgb2lms_log(const Mat& rgb, Mat& lms) {
    
    Mat mat_rgb2lms = ( Mat_<double>(3,3) <<
                       0.3811, 0.5783, 0.0402,
                       0.1967, 0.7244, 0.0782,
                       0.0241, 0.1288, 0.8444 );
    log( (mat_rgb2lms*rgb + 1.), lms );
    lms = lms / log(10);
}

void ColorTransfer::lms2lab(const Mat& lms, Mat& lab) {
    Mat mat_lms2lab = ( Mat_<double>(3,3) << 1/sqrt(3), 0, 0, 0, 1/sqrt(6), 0, 0, 0, 1/sqrt(2) )*
    ( Mat_<double>(3,3) << 1, 1, 1, 1, 1, -2, 1, -1, 0 );
    lab = mat_lms2lab*lms;
}

void ColorTransfer::lab_color_transfer(const Mat& src, const Mat& target, Mat& out) {
    Scalar src_l_mean, src_a_mean, src_b_mean, src_l_stddev, src_a_stddev, src_b_stddev,
    tar_l_mean, tar_a_mean, tar_b_mean, tar_l_stddev, tar_a_stddev, tar_b_stddev;
    
    meanStdDev(src.row(0), src_l_mean, src_l_stddev);
    meanStdDev(src.row(1), src_a_mean, src_a_stddev);
    meanStdDev(src.row(2), src_b_mean, src_b_stddev);
    
    meanStdDev(target.row(0), tar_l_mean, tar_l_stddev);
    meanStdDev(target.row(1), tar_a_mean, tar_a_stddev);
    meanStdDev(target.row(2), tar_b_mean, tar_b_stddev);
    
    Mat src_l_star = src.row(0) - src_l_mean[0];
    Mat src_a_star = src.row(1) - src_a_mean[0];
    Mat src_b_star = src.row(2) - src_b_mean[0];
    
    Mat result_l = tar_l_stddev[0]/src_l_stddev[0]*src_l_star + tar_l_mean[0];
    Mat result_a = tar_a_stddev[0]/src_a_stddev[0]*src_a_star + tar_a_mean[0];
    Mat result_b = tar_b_stddev[0]/src_b_stddev[0]*src_b_star + tar_b_mean[0];
    src_l_star.release();
    src_a_star.release();
    src_b_star.release();
    
    result_l.copyTo(out.row(0));
    result_a.copyTo(out.row(1));
    result_b.copyTo(out.row(2));
}

void ColorTransfer::lab2lms_log(const Mat& lab, Mat& lms) {
    Mat mat_lab2lms = ( Mat_<double>(3,3) << 1, 1, 1, 1, 1, -1, 1, -2, 0 )*
    ( Mat_<double>(3,3) << 1/sqrt(3), 0, 0, 0, 1/sqrt(6), 0, 0, 0, 1/sqrt(2) );
    Mat tmp;
    exp(mat_lab2lms*lab*log(10), tmp);
    lms = tmp;
}

void ColorTransfer::lms2rgb(const Mat& lms, Mat& rgb) {
    Mat mat_lms2rgb = ( Mat_<double>(3,3) <<
                       4.4679, -3.5873,  0.1193,
                       -1.2186,  2.3809, -0.1624,
                       0.0497, -0.2439,  1.2045 );
    rgb = mat_lms2rgb*lms;
}

bool ColorTransfer::run(const Mat& src, const Mat& target, Mat& result) {
    m_src = &src;
    m_target = &target;
    Mat src_double_type, tar_double_type;
    
    // 8UC3 to 64FC3
    type_convert2double(*m_src, src_double_type);
    type_convert2double(*m_target, tar_double_type);
    
    // m*n*3 to 3*(m*n)*1
    Mat src_tmp_result = Mat::zeros(3, m_src->cols*m_src->rows, CV_64FC1);
    Mat tar_tmp_result = Mat::zeros(3, m_target->cols*m_target->rows, CV_64FC1);
    rgb2vec(src_double_type, src_tmp_result);
    rgb2vec(tar_double_type, tar_tmp_result);
    src_double_type.release();
    tar_double_type.release();
    
    // RGB to LMS logarithmic space
    rgb2lms_log(src_tmp_result, src_tmp_result);
    rgb2lms_log(tar_tmp_result, tar_tmp_result);
    
    // LMS to lab
    lms2lab(src_tmp_result, src_tmp_result);
    lms2lab(tar_tmp_result, tar_tmp_result);
    
    // statistics and color transfer in lab space
    lab_color_transfer(src_tmp_result, tar_tmp_result, src_tmp_result);
    tar_tmp_result.release();
    
    // lab to LMS logarithmic space
    lab2lms_log(src_tmp_result, src_tmp_result);
    
    // LMS to RGB
    lms2rgb(src_tmp_result, src_tmp_result);
    
    // 3*(m*n)*1 to m*n*3
    vec2rgb(src_tmp_result, result);
    
    // 64FC3 to 8UC1
    type_convert2uchar(result, result);
    
    return true;
}