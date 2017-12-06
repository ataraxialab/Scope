cmake_minimum_required(VERSION 3.7)
project(Scope)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
endif()

set(test_src)
option(USE_MONGO_CXX_DRIVER "Use mongo CXX drivers" ON)
option(USE_OpenCV "Use OPENCV" ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Utils.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)


