#ifndef OUTPUT_H
#define OUTPUT_H
#include <Eigen/Dense> // #include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

// Function declarations
void writeLabelsToCSV(const std::vector<int> &labels,
                      const std::string &filename);

#endif
