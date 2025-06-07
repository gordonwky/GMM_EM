cmake_minimum_required(VERSION 3.10)
set(CMAKE_C_COMPILER /opt/homebrew/bin/gcc-14)

set(CMAKE_CXX_COMPILER /opt/homebrew/bin/g++-14)

project(MyGMM)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen library
find_package(Eigen3 REQUIRED)

# Add include directories for Eigen
include_directories(${EIGEN3_INCLUDE_DIRS})
message(STATUS "Eigen include directory: ${EIGEN3_INCLUDE_DIRS}")


# Add your source files
set(SOURCES
    GMM.cpp
    KMeans.cpp
    
)

# Add headers
set(HEADERS
    GMM.h
    KMeans.h
)

# Create a library
add_library(MyGMM ${SOURCES} ${HEADERS})

# Link Eigen library
target_link_libraries(MyGMM Eigen3::Eigen)
