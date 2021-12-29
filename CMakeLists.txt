cmake_minimum_required(VERSION 2.8)
project(nn_search)

# set(PCL_DIR /usr/lib/x86_64-linux-gnu/cmake/pcl) # pcl 1.8
set(SRC ./src)
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS}
                    /usr/include/Eigen
                    ${SRC}/)

add_definitions(${PCL_DEFINITIONS})

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -g -ggdb")

set(THIRD_PARTY_LIBS
    ${PCL_LIBRARIES})

add_executable(test_compare_with_pcl ./test/test_compare_with_pcl.cpp)
target_link_libraries(test_compare_with_pcl ${THIRD_PARTY_LIBS})