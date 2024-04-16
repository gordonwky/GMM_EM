#include <Eigen/Dense> // #include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Eigen::VectorXd;

std::vector<Eigen::VectorXd> csv_to_data(std::string file_path) {

  std::vector<Eigen::VectorXd> data;
  std::ifstream file(file_path);
  if (!file) {
    throw std::runtime_error("File not found!");
  }
  std::string line;
  while (std::getline(file, line)) {
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
      try {
        row.push_back(std::stod(cell));
      } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        throw e;
      }
    }

    VectorXd vec(row.size());
    for (int i = 0; i < row.size(); i++) {
      vec(i) = row[i];
    }
    data.push_back(vec);
  }
  if (data.size() == 0)
    throw std::invalid_argument("Dataset is empty!");
  return data;
}
