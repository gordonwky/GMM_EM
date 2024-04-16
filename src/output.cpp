#include <fstream>
#include <iostream>
#include <vector>
// Function to write the predicted labels to a CSV file
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
