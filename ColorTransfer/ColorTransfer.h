//
//  ColorTransfer.h
//  ColorTransfer
//
//  Created by Hana_Chang on 2015/5/13.
//  Copyright (c) 2015年 Andrea.C. All rights reserved.
//

#ifndef COLOR_TRANSFER_H_
#define COLOR_TRANSFER_H_

namespace cv {
    class Mat;
}

class ColorTransfer {
private:
    const cv::Mat* m_src;
    const cv::Mat* m_target;
    
    void type_convert2double(const cv::Mat& in, cv::Mat& out);
    void type_convert2uchar(const cv::Mat& in, cv::Mat& out);
    void rgb2vec(const cv::Mat& rgb3n, cv::Mat& rgb1n);
    void vec2rgb(const cv::Mat& rgb1n, cv::Mat& rgb3n);
    void rgb2lms_log(const cv::Mat& rgb, cv::Mat& lms);
    void lms2lab(const cv::Mat& lms, cv::Mat& lab);
    void lab_color_transfer(const cv::Mat& src, const cv::Mat& target, cv::Mat& out);
    void lab2lms_log(const cv::Mat& lab, cv::Mat& lms);
    void lms2rgb(const cv::Mat& lms, cv::Mat& rgb);
    
public:
    ColorTransfer();
    bool run(const cv::Mat& src, const cv::Mat& target, cv::Mat& result);
};



#endif /* COLOR_TRANSFER_H */
