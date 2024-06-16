#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

#define SQUARE_SIZE 2.25 // each square is 2.25 inches a side

const Size boardSize(7, 7);

double computeReprojectionErrors(const std::vector<std::vector<Point3f> >& objectPoints,
                                 const std::vector<std::vector<Point2f> >& imagePoints,
                                 const std::vector<Mat>& rvecs, const std::vector<Mat>& tvecs,
                                 const Mat& cameraMatrix , const Mat& distCoeffs,
                                 std::vector<float>& perViewErrors) {
    std::vector<Point2f> imagePoints2;
    size_t totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for(size_t i = 0; i < objectPoints.size(); ++i ) {
        projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
        err = norm(imagePoints[i], imagePoints2, NORM_L2);

        size_t n = objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err*err/n);
        totalErr        += err*err;
        totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);    
}

// this is mostly lifted from the opencv example
bool runCalibration(Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs, std::vector<std::vector<Point2f>> imagePoints, std::vector<Mat>& rvecs, std::vector<Mat>& tvecs,
                    std::vector<float>& reprojErrors, double& totalAvgError, std::vector<Point3f>& newObjPoints, float gridWidth) {

    cameraMatrix = Mat::eye(3, 3, CV_64F);

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    std::vector<std::vector<Point3f>> objectPoints(1);
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objectPoints[0].push_back(Point3f(j*SQUARE_SIZE, i*SQUARE_SIZE, 0));
        }
    }

    objectPoints[0][boardSize.width - 1].x = objectPoints[0][0].x + gridWidth;
    newObjPoints = objectPoints[0];

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    // this is supposed to calculate intrinsic camera parameters
    double rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, -1, cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints, CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5);

    std::cout << "Reprojection error: " << rms << std::endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    objectPoints.clear();
    objectPoints.resize(imagePoints.size(), newObjPoints);
    totalAvgError = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrors);

    return ok;
}

int main(int argc, char** argv) {
    
    if(argc < 4) {
        std::cout << "Usage: calibrate_camera <camera_id> <num_frames> <out_file>" << std::endl;
        return -1;
    }

    int cameraId = std::stoi(argv[1]);
    int numFrames = std::stoi(argv[2]);

    std::cout << "Camera ID: " << cameraId << std::endl;
    std::cout << "Number of Frames: " << numFrames << std::endl;

    VideoCapture capture(cameraId);

    if(!capture.isOpened()) {
        std::cout << "Unable to open camera with id " << cameraId << std::endl;
        return -1;
    }

    int foundCount = 0;

    std::vector<std::vector<Point2f>> imagePoints;
    Mat cameraMatrix;
    Mat distCoeffs;
    Size imageSize;
    std::vector<Mat> rvecs;
    std::vector<Mat> tvecs;
    std::vector<float> reprojErrors;
    double totalAvgError;
    std::vector<Point3f> newObjPoints;

    bool captureNext = false;

    while(foundCount < numFrames) {
        Mat frame;
        capture >> frame;

        imageSize = frame.size();

        std::vector<Point2f> pointBuf;

        bool foundChessboard = findChessboardCorners(frame, boardSize, pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        // may need some more code here, unclear from docs

        drawChessboardCorners(frame, boardSize, Mat(pointBuf), foundChessboard);

        if(foundChessboard && captureNext) {
            captureNext = false;
            foundCount++;
            std::cout << foundCount << std::endl;
            drawChessboardCorners(frame, boardSize, Mat(pointBuf), foundChessboard);
            imagePoints.push_back(pointBuf);
        }

        imshow("Chessboard", frame);

        // if(char(pollKey()) == 27) break;
        if(char(pollKey()) == 'c') captureNext = true;
    }

    runCalibration(imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrors, totalAvgError, newObjPoints, 7 * SQUARE_SIZE);

    for(;;) {
        Mat frame;
        capture >> frame;

        Mat frameUndistort = frame.clone();
        undistort(frame, frameUndistort, cameraMatrix, distCoeffs);

        imshow("Chessboard", frameUndistort);

        if(char(pollKey()) == 27) break;
    }

    capture.release();

    return 0;
}