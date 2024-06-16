cmake_minimum_required(VERSION 2.8)

project(stereo_vision)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(stereo_vision src/StereoVision.cpp)
add_executable(calibrate_camera src/Calibration.cpp)

target_link_libraries(stereo_vision ${OpenCV_LIBS})
target_link_libraries(calibrate_camera ${OpenCV_LIBS})
