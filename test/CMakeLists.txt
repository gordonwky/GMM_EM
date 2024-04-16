cmake_minimum_required(VERSION 3.0)
project(MyTestProject)

# Locate GTest package
# find_package(GTest REQUIRED)
# target_link_libraries(GTest::gtest GTest::gtest_main)
# include_directories(${GTEST_INCLUDE_DIRS})
# message(STATUS ${GTEST_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
# Add include directories for Eigen
include_directories(${EIGEN3_INCLUDE_DIRS} )

# Add your test program
add_executable(input_test Input_test.cpp)
add_executable(output_test Output_test.cpp)
# Link against GTest and pthread library
find_package(GTest REQUIRED)
# target_link_libraries(test_my_program ${GTEST_BOTH_LIBRARIES} pthread)
target_link_libraries( input_test GTest::gtest GTest::gtest_main) 
target_link_libraries( output_test GTest::gtest GTest::gtest_main) 

# target_link_libraries(test_my_program PRIVATE gtest gtest_main pthread)

enable_testing()
add_test(iostest1 input_test)

add_test(iostest2 output_test)