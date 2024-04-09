cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add include directories for Eigen

# Find Eigen library
find_package(Eigen3 REQUIRED)

# Add include directories for Eigen
include_directories(${EIGEN3_INCLUDE_DIRS})

# include_directories(/path/to/eigen-3.4.0)

# Add your main executable
add_executable(my_test main.cpp)

link_directories(/Users/kimyingwong/GMM_EM/src/build)
# Link against your static library
target_link_libraries(my_test libMyGMM.a)
