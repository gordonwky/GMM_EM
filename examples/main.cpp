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

  std::vector<VectorXd> data = csv_to_data("../dataset/two_gaussian.csv.csv");

  GMM gmm(3, 100, 1e-2, "kmeans");
  // gmm.initiate(data);
  auto start = std::chrono::high_resolution_clock::now();
  gmm.fit(data);
  // printf("Means: \n");
  // for (const auto &m : gmm.mean_vector) {
  //   std::cout << m.transpose() << std::endl;
  // }
  // for (const auto &m : gmm.cov_matrices) {
  //   std::cout << m << std::endl;
  // }
  // printf("normal: \n");
  // gmm.computing_normal_distribution_pdf(data);
  // MatrixXd normal = gmm.normal_distribution_pdf;
  // std::cout << normal << std::endl;
  std::vector<int> label = gmm.predict();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  for (const auto &l : label) {
    std::cout << l << std::endl;
  }
  printf("Time: %f\n", elapsed.count());
  writeLabelsToCSV(label, "../dataset/gaussian_predict_label.csv");

  return 0;
}