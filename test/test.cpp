// #include "GMM.h"
// #include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
// using Eigen::VectorXd;
// g++ test.cpp -o a.out -std=c++11 -lgtest -lpthread

int add(int a, int b) { return a + b; }

TEST(AddTest, TestAddition) {
  EXPECT_EQ(add(2, 3), 5);
  EXPECT_EQ(add(-1, 1), 0);
  EXPECT_EQ(add(0, 0), 0);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}