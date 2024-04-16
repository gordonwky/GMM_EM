#include <fstream>
#include <iostream>
#include <vector>
// Function to write the predicted labels to a CSV file
void writeLabelsToCSV(const std::vector<int> &labels,
                      const std::string &filename) {
  // file name contains the csv at the end
  if (filename.size() <= 4 || filename.substr(filename.size() - 4) != ".csv") {
    throw std::invalid_argument(
        "Invalid file extension. Please provide a CSV file.");
  }
  std::ofstream file(filename);
  if (file.is_open()) {
    file << "labels" << "," << "\n";
    for (const auto &l : labels) {
      file << l << "," << "\n";
    }
    file.close();
    std::cout << "Labels saved to labels.csv" << std::endl;
  } else {
    throw std::invalid_argument("Unable to open file: " + filename);
    // std::cerr << "Unable to open file: " << filename << std::endl;
  }
}
