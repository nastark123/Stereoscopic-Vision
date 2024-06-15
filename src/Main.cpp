#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv) {
    VideoCapture capLeft(2);
    VideoCapture capRight(0);

    if(!capLeft.isOpened() || !capRight.isOpened()) {
        printf("Failed to open video\n");
        return -1;
    }

    namedWindow("Display Left", WINDOW_AUTOSIZE);
    namedWindow("Display Right", WINDOW_AUTOSIZE);
    namedWindow("Disparity", WINDOW_AUTOSIZE);

    Ptr<StereoBM> stereo = StereoBM::create(128, 11);
    (*stereo).setTextureThreshold(15);
    (*stereo).setSpeckleRange(0);

    for(;;) {
        Mat leftFrame;
        Mat rightFrame;
        capLeft >> leftFrame;
        capRight >> rightFrame;

        cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);
        cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);

        Mat disparity;
        (*stereo).compute(leftFrame, rightFrame, disparity);

        Mat disparityCvt;
        disparity.convertTo(disparityCvt, CV_32F, 1.0);
        disparityCvt = (disparityCvt/16.0f - 0.0f / 64.0f);

        imshow("Display Left", leftFrame);
        imshow("Display Right", rightFrame);
        imshow("Disparity", disparityCvt);
        if(pollKey() == 1048603) break;
    }

    capLeft.release();
    capRight.release();

    return 0;
}
