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

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path/to/data.csv>" << std::endl;
    return 1;
  }

  std::string csv_path = argv[1];

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<Eigen::VectorXd> data = csv_to_data(csv_path);

  KMeans kmeans(2, 100);
  kmeans.Initiate(data);
  kmeans.fit(data);

  // GMM gmm(2, 100, 1e-2, "kmeans");
  // gmm.fit(data);
  // std::vector<int> label = gmm.predict();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  // writeLabelsToCSV(label, "output.csv");

  return 0;
}