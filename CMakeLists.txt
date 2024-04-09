cmake_minimum_required(VERSION 3.10)
project(MyGMM)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Eigen library
find_package(Eigen3 REQUIRED)

# Add include directories for Eigen
include_directories(${EIGEN3_INCLUDE_DIRS})
# message(STATUS "Eigen include directory: ${EIGEN3_INCLUDE_DIRS}")


# Add your source files
SET (PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
message(STATUS "project: ${PROJECT_SOURCE_DIR}")

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




# Add your main executable
add_executable(my_test main.cpp)

# Link against your static library
target_link_libraries(my_test libMyGMM.a)
