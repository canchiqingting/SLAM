cmake_minimum_required( VERSION 2.8 )

project( feature_Extraction )

set( CMAKE_CXX_FLAGS "-std=c++11" )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( feature_Extraction feature_extraction.cpp )

target_link_libraries( feature_Extraction ${OpenCV_LIBS} )
