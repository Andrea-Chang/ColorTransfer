//
//  main.cpp
//  ColorTransfer
//
//  Created by Hana_Chang on 2015/5/13.
//  Copyright (c) 2015年 Andrea.C. All rights reserved.
//

#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "ColorTransfer.h"

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {

    // image path
    string src_filename    = "/Users/Hana/Programming/Xcode/ColorTransfer/ColorTransfer/src.jpg";
    string target_filename = "/Users/Hana/Programming/Xcode/ColorTransfer/ColorTransfer/target.jpg";
    Mat src = imread(src_filename, 1);
    Mat target = imread(target_filename, 1);
    Mat result;
    
    // perform color transfer
    ColorTransfer CT;
    CT.run(src, target, result);
    
    // write the result
    imwrite("/Users/Hana/Programming/Xcode/ColorTransfer/ColorTransfer/result.jpg", result);

    return 0;
}