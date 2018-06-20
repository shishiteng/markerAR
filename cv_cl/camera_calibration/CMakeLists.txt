cmake_minimum_required(VERSION 2.6)
project(camera_calibration)

find_package(OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIR})

link_directories(${OpenCV_LIBRARY_DIR})

include_directories(${cvCalib_SOURCE_DIR}/src)
#add_subdirectory(src)

add_executable(camera_calibration 
    camera_calibration.cpp
    )

target_link_libraries( camera_calibration ${OpenCV_LIBRARIES} )
     
install (TARGETS camera_calibration DESTINATION bin)


