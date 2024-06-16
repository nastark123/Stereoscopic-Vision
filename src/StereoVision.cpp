#include <stdio.h>
#include <stdint.h>
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
    namedWindow("Disparity Colors", WINDOW_AUTOSIZE);

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

        printf("%d\n", disparity.type());

        int16_t min = 32767;
        int16_t max = -32768;
        long avg = 0;

        // Mat color(disparity.rows, disparity.cols, CV_8UC3);

        // for (int16_t p : cv::Mat_<int16_t>(disparity)) {
        //     if(p < min) min = p;
        //     if(p > max) max = p;
        //     avg += p;
        // }

        // for(int i = 0; i < disparity.rows; i++) {
        //     for(int j = 0; j < disparity.cols; j++) {
        //         int16_t grayColor = disparity.at<int16_t>(i, j);
        //         if(grayColor < min) min = grayColor;
        //         if(grayColor > max) max = grayColor;
        //         avg += grayColor;
        //         if(grayColor < -16383) {
        //             color.at<Vec3b>(i, j)[0] = 0;
        //             color.at<Vec3b>(i, j)[1] = 0;
        //             color.at<Vec3b>(i, j)[2] = ((32766 + grayColor) / 64);
        //         } else if(grayColor < 0) {
        //             color.at<Vec3b>(i, j)[0] = 0;
        //             color.at<Vec3b>(i, j)[1] = ((16383 + grayColor) / 64);
        //             color.at<Vec3b>(i, j)[2] = ((16383 + grayColor) / 64);
        //         } else if(grayColor < 16384) {
        //             color.at<Vec3b>(i, j)[0] = 0;
        //             color.at<Vec3b>(i, j)[1] = ((grayColor) / 64);
        //             color.at<Vec3b>(i, j)[2] = 255 - (grayColor / 64);
        //         } else {
        //             color.at<Vec3b>(i, j)[0] = ((grayColor - 16384) / 64);
        //             color.at<Vec3b>(i, j)[1] = ((grayColor - 16384) / 64);
        //             color.at<Vec3b>(i, j)[2] = 0;
        //         }
        //     }
        // }

        // avg /= 640 * 480;

        // printf("Min: %d, Max: %d, Avg: %d\n", min, max, avg);

        // Mat disparityCvt;
        // disparity.convertTo(disparityCvt, CV_8U, 10.0f);
        // disparityCvt = (disparityCvt/4.0f);

        imshow("Display Left", leftFrame);
        imshow("Display Right", rightFrame);
        imshow("Disparity", disparity);
        // imshow("Disparity Colors", color);
        if(pollKey() == 1048603) break;
    }

    capLeft.release();
    capRight.release();

    return 0;
}
