cmake_minimum_required(VERSION 3.10)
project(MyGMM)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Wpedantic -g -O2 )

# Find Eigen library
find_package(Eigen3 REQUIRED)

# Add include directories for Eigen
include_directories(${EIGEN3_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

set(SOURCES ${PROJECT_SOURCE_DIR}/src/GMM.cpp ${PROJECT_SOURCE_DIR}/src/KMeans.cpp ${PROJECT_SOURCE_DIR}/src/input.cpp ${PROJECT_SOURCE_DIR}/src/output.cpp)
set(HEADERS, ${PROJECT_SOURCE_DIR}/include/GMM.h ${PROJECT_SOURCE_DIR}/include/KMeans.h  ${PROJECT_SOURCE_DIR}/src/input.h ${PROJECT_SOURCE_DIR}/src/output.h)

# 

# Create a library
add_library(MyGMM ${SOURCES} ${HEADERS})

# Link Eigen library
target_link_libraries(MyGMM Eigen3::Eigen)

# Add your main executable

add_executable(my_example ${PROJECT_SOURCE_DIR}/examples/main.cpp)

target_link_libraries(my_example MyGMM)

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)

# Link the library Gtest
# find_package(GTest CONFIG REQUIRED) 
add_executable(test_gmm ${PROJECT_SOURCE_DIR}/test/gmm_test.cpp)
add_executable(test_input ${PROJECT_SOURCE_DIR}/test/Input_test.cpp)
add_executable(test_output ${PROJECT_SOURCE_DIR}/test/Output_test.cpp)

target_link_libraries( test_gmm gtest gtest_main MyGMM) 
target_link_libraries( test_input gtest gtest_main MyGMM) 
target_link_libraries( test_output gtest gtest_main MyGMM) 

# # target_link_libraries( test_gmm GTest::gtest GTest::gtest_main) 
# # include_directories(${GTEST_INCLUDE_DIRS})

enable_testing()
add_test(test_gmm_main test_gmm)
add_test(test_gmm_input test_input)
add_test(test_gmm_output test_output)


# # Add test executable


# target_link_libraries(test_gmm MyGMM ${GTEST_BOTH_LIBRARIES} pthread)


