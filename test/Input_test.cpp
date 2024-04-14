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

TEST(CSVToDataTest, ValidCSV) {
  // Provide a path to your test CSV file
  std::string file_path = "../inputTest1.csv";

  std::vector<Eigen::VectorXd> expected_data = {
      Eigen::VectorXd::Ones(3), Eigen::VectorXd::Ones(3),
      Eigen::VectorXd::Ones(3)}; // Define expected output data

  // Call the function
  std::vector<Eigen::VectorXd> actual_data = csv_to_data(file_path);

  // Assert the actual output matches the expected output
  ASSERT_EQ(actual_data.size(), expected_data.size());
  for (size_t i = 0; i < actual_data.size(); ++i) {
    ASSERT_EQ(actual_data[i].size(), expected_data[i].size());
    for (int j = 0; j < actual_data[i].size(); ++j) {
      ASSERT_DOUBLE_EQ(actual_data[i](j), expected_data[i](j));
    }
  }
}

TEST(CSVToDataTest, InvalidCSV) {
  // Provide a path to your test CSV file
  std::string file_path = "../xxx.csv";
  EXPECT_THROW(csv_to_data(file_path), std::runtime_error);
}

TEST(CSVToDataTest, InValidValuesCSV) {
  // Provide a path to your test CSV file
  std::string file_path = "../inputTest2.csv";
  EXPECT_THROW(csv_to_data(file_path), std::invalid_argument);
}

TEST(CSVToDataTest, EmptyCSV) {
  // Provide a path to your test CSV file
  std::string file_path = "../empty.csv";
  EXPECT_THROW(csv_to_data(file_path), std::invalid_argument);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
