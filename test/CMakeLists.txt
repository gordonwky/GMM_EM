cmake_minimum_required(VERSION 3.0)
project(MyTestProject)

# Locate GTest package
# find_package(GTest REQUIRED)
# target_link_libraries(GTest::gtest GTest::gtest_main)
# include_directories(${GTEST_INCLUDE_DIRS})
# message(STATUS ${GTEST_INCLUDE_DIRS})

# Add your test program
add_executable(test_my_program test.cpp)

# Link against GTest and pthread library
find_package(GTest REQUIRED)
# target_link_libraries(test_my_program ${GTEST_BOTH_LIBRARIES} pthread)
message(STATUS ${PROJECT_SOURCE_DIR})
target_link_libraries( test_my_program GTest::gtest GTest::gtest_main) 

# target_link_libraries(test_my_program PRIVATE gtest gtest_main pthread)

enable_testing()
add_test(test_arithmetic1 test_my_program)
