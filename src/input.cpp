#include <Eigen/Dense> // #include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>
using Eigen::VectorXd;

std::vector<Eigen::VectorXd> csv_to_data(std::string file_path) {
  try {
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
        }
      }

      VectorXd vec(row.size());
      for (int i = 0; i < row.size(); i++) {
        vec(i) = row[i];
      }
      data.push_back(vec);
    }
    if (data.size() == 0)
      throw "Dataset is empty!";
    return data;
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    // Handle the exception as needed.
  }
}
