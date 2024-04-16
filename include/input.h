#ifndef INPUT_H
#define INPUT_H
#include <Eigen/Dense> // #include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>
// Function declarations
std::vector<Eigen::VectorXd> csv_to_data(std::string file_path);
#endif