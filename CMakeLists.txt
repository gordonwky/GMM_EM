cmake_minimum_required(VERSION 3.10)
project(MyGMM)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Wpedantic -g -O2)

# Find Eigen library
find_package(Eigen3 REQUIRED)

# Add include directories for Eigen
include_directories(${EIGEN3_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

set(SOURCES ${PROJECT_SOURCE_DIR}/src/GMM.cpp ${PROJECT_SOURCE_DIR}/src/KMeans.cpp)
set(HEADERS, ${PROJECT_SOURCE_DIR}/include/GMM.h ${PROJECT_SOURCE_DIR}/include/KMeans.h)

#message(STATUS ${PROJECT_SOURCE_DIR})

# Create a library
add_library(MyGMM ${SOURCES} ${HEADERS})

# Link Eigen library
target_link_libraries(MyGMM Eigen3::Eigen)

# Add your main executable
add_executable(my_test ${PROJECT_SOURCE_DIR}/examples/main.cpp)
target_link_libraries(my_test MyGMM)
