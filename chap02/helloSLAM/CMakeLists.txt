cmake_minimum_required(VERSION 2.8)

project(HelloSLAM)

add_library(hello SHARED libHelloSLAM.cpp)
add_executable(useHello useHello.cpp)
target_link_libraries(useHello hello)
