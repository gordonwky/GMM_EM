#include "KMeans.h" // Assuming you have a kmeans.h header file with the k-means implementation
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

TEST(KMeansTest, TwoClusters) {
  std::vector<VectorXd> data;
  data.push_back(VectorXd(2));
  data[0] << 1.0, 2.0;
  data.push_back(VectorXd(2));
  data[1] << 3.0, 4.0;
  data.push_back(VectorXd(2));
  data[2] << 5.0, 6.0;
  data.push_back(VectorXd(2));
  data[3] << 7.0, 8.0;
  //   int k = 2;
  KMeans kmeans(2, 100);
  kmeans.fit(data);

  std::vector<VectorXd> expectedCentroids;
  expectedCentroids.push_back(VectorXd(2));
  expectedCentroids[0] << 3.0, 4.0;
  expectedCentroids.push_back(VectorXd(2));
  expectedCentroids[1] << 7.0, 8.0;
  std::vector<VectorXd> actualCentroids = kmeans.get_mean_vector();

  EXPECT_EQ(expectedCentroids.size(), actualCentroids.size());
  for (int i = 0; i < expectedCentroids.size(); i++) {
    EXPECT_EQ(expectedCentroids[i], actualCentroids[i]);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
