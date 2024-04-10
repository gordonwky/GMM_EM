#include "GMM.h"
#include <gtest/gtest.h>
#include <iostream>

TEST(GMMTest, Initiate) {
  GMM gmm(3, 100, 1e-2, "kmeans");
  std::vector<VectorXd> data;
  data.push_back(VectorXd(2));
  data[0] << 1, 2;
  data.push_back(VectorXd(2));
  data[1] << 2, 3;
  data.push_back(VectorXd(2));
  data[2] << 3, 4;
  data.push_back(VectorXd(2));
  data[3] << 4, 5;
  data.push_back(VectorXd(2));
  data[4] << 5, 6;
  data.push_back(VectorXd(2));
  data[5] << 10, 11;
  data.push_back(VectorXd(2));
  data[6] << 11, 12;
  gmm.initiate(data);
  EXPECT_EQ(gmm.responbilities.rows(), 7);
  EXPECT_EQ(gmm.responbilities.cols(), 3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}