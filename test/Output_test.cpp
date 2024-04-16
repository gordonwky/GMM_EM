#include "output.h"
#include <Eigen/Dense>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Define a test fixture
// g++ -std=c++17  my_executable Input_test.cpp -L../build/ -lMyGMM
// -I/path/to/eigen-3.4.0 Test case for csv_to_data function

TEST(WriteLabelsToCSVTest, ValidLabels) {
  // Provide a path to your test CSV file
  std::string file_path = "../test_labels.csv";
  std::vector<int> labels = {1, 2, 3, 4, 5};
  writeLabelsToCSV(labels, file_path);
  std::ifstream file(file_path);
  std::string line;
  std::getline(file, line);
  EXPECT_EQ(line, "labels,");
  for (const auto &l : labels) {
    std::getline(file, line);
    EXPECT_EQ(std::stoi(line), l);
  }
}

TEST(WriteLabelsToCSVTest, InvalidPath) {
  // Provide a path to your test CSV file
  std::string file_path = "../dsds";
  std::vector<int> labels = {1, 2, 3, 4, 5};
  EXPECT_THROW(writeLabelsToCSV(labels, file_path), std::invalid_argument);
}
