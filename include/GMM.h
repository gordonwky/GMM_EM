#ifndef GMM_H
#define GMM_H
#include "KMeans.h"
#include <Eigen/Dense> // #include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class GMM {
private:
  int re_initialize_count = 50;
  int numComponents;
  int max_iter;
  double tol;
  int numData;
  int dimData;
  bool fitted = false;
  std::string init_method = "random";
  // double likelihood = 1e30;

public:
  double likelihood = 1e30;
  GMM();
  GMM(int numComponents, int max_iter, double tol, std::string init_method);

  MatrixXd responbilities;
  MatrixXd normal_distribution_pdf;        // normal distribution pdf
  std::vector<VectorXd> mean_vector;       // mean
  std::vector<double> mixing_coefficients; // weight
  std::vector<MatrixXd> cov_matrices;      // cov

  void initiate(const std::vector<VectorXd> &data);
  void computing_normal_distribution_pdf(const std::vector<VectorXd> &data);

  // MatrixXd computing_normal_distribution_pdf(const std::vector<VectorXd>
  // &data); double GMM::compute_log_likelihood(const std::vector<VectorXd>
  // &data);
  void E_step();
  void M_step(const std::vector<VectorXd> &data);
  void fit(const std::vector<VectorXd> &data);
  double compute_log_likelihood(const std::vector<VectorXd> &data);
  std::vector<int> predict();
  // int predict(const VectorXd &single_point);
};
#endif