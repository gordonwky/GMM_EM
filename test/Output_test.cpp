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
void writeLabelsToCSV(const std::vector<int> &labels,
                      const std::string &filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    file << "labels" << "," << "\n";
    for (const auto &l : labels) {
      file << l << "," << "\n";
    }
    file.close();
    std::cout << "Labels saved to labels.csv" << std::endl;
  } else {
    std::cerr << "Unable to open file: " << filename << std::endl;
  }
}

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

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
