#include "GMM.h"
#include "input.h"
#include "output.h"
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {

  // Test for System Test 1
  std::vector<VectorXd> data = csv_to_data("../dataset/two_gaussian.csv");

  GMM gmm(2, 100, 1e-2, "kmeans");

  gmm.fit(data);

  std::vector<int> label = gmm.predict();

  writeLabelsToCSV(label, "../dataset/two_gaussian_predict_label.csv");

  return 0;
}