cmake_minimum_required(VERSION 2.8)
project(nn_search)

# set(PCL_DIR /usr/lib/x86_64-linux-gnu/cmake/pcl) # pcl 1.8
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(/usr/include/Eigen)
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -g -ggdb")

set(SRC ./src)
set(THIRD_PARTY_LIBS
    ${PCL_LIBRARIES})

add_executable(main ${SRC}/main.cpp)
target_link_libraries(main ${THIRD_PARTY_LIBS})