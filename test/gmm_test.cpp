#include "GMM.h"
#include "input.h"
#include <Eigen/Dense>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

double RandIndex(const std::vector<int> &clustering1,
                 const std::vector<int> &clustering2) {
  size_t n = clustering1.size();

  int a = 0, b = 0;

  // Compare each pair of elements
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (clustering1[i] == clustering1[j] &&
          clustering2[i] == clustering2[j]) {
        // Both clusterings group the pair in the same cluster
        ++a;
      } else if (clustering1[i] != clustering1[j] &&
                 clustering2[i] != clustering2[j]) {
        // Both clusterings group the pair in different clusters
        ++b;
      }
      // If they are in different clusters in one clustering and in the same
      // cluster in the other, we don't need to do anything, as it does not
      // contribute to a or b.
    }
  }

  // Calculate Rand Index
  double randIndex = static_cast<double>(a + b) / (n * (n - 1) / 2);
  return randIndex;
}

TEST(GMMTest, IrisData) {
  std::vector<VectorXd> data = csv_to_data("../dataset/iris.csv");
  GMM gmm(3, 100, 1e-2, "kmeans");
  gmm.fit(data);
  std::vector<int> label = gmm.predict();

  std::vector<int> sklearn_label;
  std::ifstream file("../dataset/sklearn_iris_label.csv");
  if (!file) {
    throw std::runtime_error("File not found!");
  }
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      try {
        sklearn_label.push_back(
            std::stoi(cell)); // Cast cell value to an integer
      } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        throw e;
      }
    }
  }
  double randIndex = RandIndex(label, sklearn_label);
  printf("Rand Index: %f\n", randIndex);
  EXPECT_GT(randIndex, 0.7);
}

TEST(GMMTest, BreastData) {
  std::vector<VectorXd> data = csv_to_data("../dataset/scaled_breast_X.csv");
  GMM gmm(3, 100, 1e-2, "kmeans");
  gmm.fit(data);
  std::vector<int> label = gmm.predict();

  std::vector<int> sklearn_label;
  std::ifstream file("../dataset/sklearn_breast_label.csv");
  if (!file) {
    throw std::runtime_error("File not found!");
  }
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      try {
        sklearn_label.push_back(
            std::stoi(cell)); // Cast cell value to an integer
      } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        throw e;
      }
    }
  }
  double randIndex = RandIndex(label, sklearn_label);
  printf("Rand Index: %f\n", randIndex);
  EXPECT_GT(randIndex, 0.7);
}

TEST(Likelihood, LikelihoodTest) {
  std::vector<VectorXd> data;
  data.push_back(VectorXd(1));
  data[0] << 1.0;
  data.push_back(VectorXd(1));
  data[1] << 2.0;
  data.push_back(VectorXd(1));
  data[2] << 3.0;
  data.push_back(VectorXd(1));
  data[3] << 4.0;
  data.push_back(VectorXd(1));
  data[4] << 5.0;

  GMM gmm(1, 100, 1e-2, "kmeans");
  gmm.fit(data);
  double likelihood = gmm.compute_log_likelihood(data);
  printf("Likelihood: %f\n", likelihood);

  EXPECT_LE(abs(likelihood - (-8.827560617423226)), 1e-2);
}

TEST(NormalDistribution, NormalDistributionTest) {
  std::vector<VectorXd> data;
  data.push_back(VectorXd(1));
  data[0] << 1.0;
  data.push_back(VectorXd(1));
  data[1] << 2.0;
  data.push_back(VectorXd(1));
  data[2] << 3.0;
  GMM gmm(1, 100, 1e-2, "kmeans");
  gmm.fit(data);
  std::cout << "Mean: " << gmm.mean_vector[0] << std::endl;
  std::cout << "\n Cov: " << gmm.cov_matrices[0] << std::endl;
  gmm.computing_normal_distribution_pdf(data);
  std::cout << "Normal Distribution PDF: " << gmm.normal_distribution_pdf
            << std::endl;
  std::vector<double> expected_output = {0.23079948, 0.48860251, 0.23079948};
  std::cout << gmm.normal_distribution_pdf.cols()
            << gmm.normal_distribution_pdf.rows() << std::endl;
  for (size_t i = 0; i < 3; i++) {
    printf("Expected: %f, Actual: %f\n", expected_output[i],
           gmm.normal_distribution_pdf(i, 0));
    EXPECT_LE(abs(gmm.normal_distribution_pdf(i, 0) - expected_output[i]),
              1e-2);
  }
}

TEST(Prediction, NoFitPrediction) {
  GMM gmm(1, 100, 1e-2, "kmeans");
  EXPECT_THROW(gmm.predict(), std::runtime_error);
}

TEST(Prediction, PostiveIntPrediction) {
  int numComponents = 3;
  GMM gmm(numComponents, 100, 1e-2, "kmeans");
  std::vector<VectorXd> data;
  data.push_back(VectorXd(1));
  data[0] << 1.0;
  data.push_back(VectorXd(1));
  data[1] << 2.0;
  data.push_back(VectorXd(1));
  data[2] << 3.0;
  data.push_back(VectorXd(1));
  data[3] << 4.0;
  gmm.fit(data);
  std::vector<int> labels = gmm.predict();
  for (size_t i = 0; i < labels.size(); i++) {
    // std::cout << labels[i] << std::endl;
    EXPECT_GT(labels[i], 0);
  }
  std::set<int> uniqueElements(labels.begin(), labels.end());
  EXPECT_EQ(numComponents, uniqueElements.size());
}

TEST(Prediction, ClusterEqGtDateSize) {
  // Test if the number of clusters is greater than or equal to the number of
  // data points
  int numComponents = 4;
  GMM gmm(numComponents, 100, 1e-2, "kmeans");
  std::vector<VectorXd> data;
  data.push_back(VectorXd(1));
  data[0] << 1.0;
  data.push_back(VectorXd(1));
  data[1] << 2.0;
  data.push_back(VectorXd(1));
  data[2] << 3.0;
  data.push_back(VectorXd(1));
  data[3] << 4.0;

  EXPECT_THROW(gmm.fit(data), std::runtime_error);
}