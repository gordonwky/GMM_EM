#include "GMM.h"
#include "input.h"
#include "output.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
  // std::vector<VectorXd> data;
  // data.push_back(VectorXd(2));
  // data[0] << 1, 2;
  // data.push_back(VectorXd(2));
  // data[1] << 2, 3;
  // data.push_back(VectorXd(2));
  // data[2] << 3, 4;
  // data.push_back(VectorXd(2));
  // data[3] << 4, 5;
  // data.push_back(VectorXd(2));
  // data[4] << 5, 6;
  // data.push_back(VectorXd(2));
  // data[5] << 10, 11;
  // data.push_back(VectorXd(2));
  // data[6] << 11, 12;
  // std::vector<VectorXd> data;

  std::vector<VectorXd> data = csv_to_data("../dataset/iris.csv");

  GMM gmm(3, 100, 1e-2, "kmeans");
  // gmm.initiate(data);
  gmm.fit(data);

  std::vector<int> label = gmm.predict();
  writeLabelsToCSV(label, "../test_lables.csv");

  return 0;
}