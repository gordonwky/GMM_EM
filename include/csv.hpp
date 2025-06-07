#ifndef CSV_HPP
#define CSV_HPP

#include <Eigen/Dense>
#include <stdexcept>
#include <string>
class csv_block { /* data */
private:
  Eigen::MatrixXd data;
  std::vector<std::string> header;
  std::string filename;
  bool header_exist = false;
  int nrows;
  int ncols;

public:
  csv_block();
  csv_block(const std::string &);
  void read_csv(const std::string &, bool _header_exist = false);
  void print();
};

csv_block::read_csv(const std::string &filename, bool _header_exist) {

  header_exist = _header_exist;
  std::ifstream std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("File not found!");
  }

  std::string line;
  std::getline(file, line);
  std::stringstream ss(line);
  std::string cell;
  if (header_exist) {
    while (std::getline(ss, cell, ',')) {
      header.push_back(cell);
    }
  } else {
    while (std::getline(ss, cell, ',')) {
      try {
        row.push_back(std::stod(cell));
      } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        throw e;
      }
    }
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
      throw std::invalid_argument("Warning: Dataset is empty!");
    return data;
  }
};

csv_block::csv_block() { data = nullptr; }
csv_block::csv_block(const std::string &filename) {
  this->filename = filename;
  read_csv(filename);
}
csv_block::print() { std::cout << "Data: " << data << std::endl; }

#endif
