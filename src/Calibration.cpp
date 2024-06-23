#include <ctime>
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

bool runStereoCalibration(Size& imageSize, Mat& cameraMatrix1, Mat& distCoeffs1, std::vector<std::vector<Point2f>> imagePoints1,
                            Mat& cameraMatrix2, Mat& distCoeffs2, std::vector<std::vector<Point2f>> imagePoints2, Mat& rotation, Mat& translation,
                            Mat& essential, Mat& fundamental, std::vector<Point3f> newObjPoints, float gridWidth) {

    cameraMatrix1 = Mat::eye(3, 3, CV_64F);
    cameraMatrix2 = Mat::eye(3, 3, CV_64F);

    distCoeffs1 = Mat::zeros(8, 1, CV_64F);
    distCoeffs2 = Mat::zeros(8, 1, CV_64F);

    std::vector<std::vector<Point3f>> objectPoints(1);
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            objectPoints[0].push_back(Point3f(j*SQUARE_SIZE, i*SQUARE_SIZE, 0));
        }
    }

    objectPoints[0][boardSize.width - 1].x = objectPoints[0][0].x + gridWidth;
    newObjPoints = objectPoints[0];

    objectPoints.resize(imagePoints1.size(), objectPoints[0]);

    // this is supposed to calculate intrinsic camera parameters
    double rms = stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                    imageSize, rotation, translation, essential, fundamental);

    std::cout << "Reprojection Error: " << rms << std::endl;

    bool ok = checkRange(cameraMatrix1) && checkRange(distCoeffs1) && checkRange(cameraMatrix2) && checkRange(distCoeffs2);

    objectPoints.clear();
    objectPoints.resize(imagePoints1.size(), newObjPoints);

    return ok;
}

void saveCalibration(std::string filename, Mat& cameraMatrix1, Mat& distCoeffs1, double totalAvgError1, Mat& cameraMatrix2, Mat& distCoeffs2, double totalAvgError2,
                        Mat& rotation, Mat& translation, Mat& essential, Mat& fundamental) {
    FileStorage fs(filename, FileStorage::WRITE);

    time_t tm;
    time(&tm);
    struct tm *t2 = localtime(&tm);
    char buf[1024];
    strftime(buf, sizeof(buf), "%c", t2);
    fs << "calibration_time" << buf;

    fs << "camera_matrix1" << cameraMatrix1;
    fs << "distortion_coefficients1" << distCoeffs1;
    fs << "avg_reprojection_error1" << totalAvgError1;
    fs << "camera_matrix2" << cameraMatrix2;
    fs << "distortion_coeffecients2" << distCoeffs2;
    fs << "avg_reprojection_error2" << totalAvgError2;
    fs << "rotation_matrix" << rotation;
    fs << "translation_matrix" << translation;
    fs << "essential_matrix" << essential;
    fs << "fundamental_matrix" << fundamental;
}

int main(int argc, char** argv) {
    
    if(argc < 4) {
        std::cout << "Usage: calibrate_camera <camera_id_1> <camera_id_2> <num_frames> <out_file>" << std::endl;
        return -1;
    }

    int cameraId1 = std::stoi(argv[1]);
    int cameraId2 = std::stoi(argv[2]);
    int numFrames = std::stoi(argv[3]);

    std::cout << "Camera ID 1: " << cameraId1 << std::endl;
    std::cout << "Camera ID 2: " << cameraId2 << std::endl;
    std::cout << "Number of Frames: " << numFrames << std::endl;

    VideoCapture capture1(cameraId1);
    VideoCapture capture2(cameraId2);

    if(!capture1.isOpened()) {
        std::cout << "Unable to open camera with id " << cameraId1 << std::endl;
        return -1;
    }

    if(!capture2.isOpened()) {
        std::cout << "Unable to open camera with id " << cameraId2 << std::endl;
        return -1;
    }

    int foundCount = 0;

    std::vector<std::vector<Point2f>> imagePoints1;
    Mat cameraMatrix1;
    Mat distCoeffs1;
    std::vector<std::vector<Point2f>> imagePoints2;
    Mat cameraMatrix2;
    Mat distCoeffs2;
    Size imageSize;
    std::vector<Mat> rvecs;
    std::vector<Mat> tvecs;
    std::vector<float> reprojErrors;
    double totalAvgError1;
    double totalAvgError2;
    std::vector<Point3f> newObjPoints;

    bool captureNext = false;

    std::cout << "Capturing frames for Camera 1..." << std::endl;
    while(foundCount < numFrames) {
        Mat frame;
        capture1 >> frame;

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
            imagePoints1.push_back(pointBuf);
        }

        imshow("Chessboard", frame);

        // if(char(pollKey()) == 27) break;
        if(char(pollKey()) == 'c') captureNext = true;
    }

    std::cout << "Finished capture for Camera 1, calibrating Camera 2..." << std::endl;

    foundCount = 0;

    while(foundCount < numFrames) {
        Mat frame;
        capture2 >> frame;

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
            imagePoints2.push_back(pointBuf);
        }

        imshow("Chessboard", frame);

        // if(char(pollKey()) == 27) break;
        if(char(pollKey()) == 'c') captureNext = true;
    }

    std::cout << "Finished capture for both cameras, running calibration and exporting..." << std::endl;

    runCalibration(imageSize, cameraMatrix1, distCoeffs1, imagePoints1, rvecs, tvecs, reprojErrors, totalAvgError1, newObjPoints, 7 * SQUARE_SIZE);
    runCalibration(imageSize, cameraMatrix2, distCoeffs2, imagePoints2, rvecs, tvecs, reprojErrors, totalAvgError2, newObjPoints, 7 * SQUARE_SIZE);

    Mat rotation;
    Mat translation;
    Mat essential;
    Mat fundamental;

    runStereoCalibration(imageSize, cameraMatrix1, distCoeffs1, imagePoints1, cameraMatrix2, distCoeffs2, imagePoints2, rotation, translation, essential, fundamental, newObjPoints, 7 * SQUARE_SIZE);

    saveCalibration(std::string(argv[3]), cameraMatrix1, distCoeffs1, totalAvgError1, cameraMatrix2, distCoeffs2, totalAvgError2, rotation, translation, essential, fundamental);

    for(;;) {
        Mat frame1;
        Mat frame2;
        capture1 >> frame1;
        capture2 >> frame2;

        Mat frameUndistort1 = frame1.clone();
        Mat frameUndistort2 = frame2.clone();
        undistort(frame1, frameUndistort1, cameraMatrix1, distCoeffs1);
        undistort(frame2, frameUndistort2, cameraMatrix2, distCoeffs2);

        imshow("Undistort 1", frameUndistort1);
        imshow("Undistort 2", frameUndistort2);

        if(char(pollKey()) == 27) break;
    }

    capture1.release();
    capture2.release();

    return 0;
}