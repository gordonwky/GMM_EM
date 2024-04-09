cmake_minimum_required(VERSION 3.0)
project(MyTestProject)

# Locate GTest package
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add your test program
add_executable(test_my_program main.cpp)

# Link against GTest and pthread library
target_link_libraries(test_my_program ${GTEST_BOTH_LIBRARIES} pthread)
