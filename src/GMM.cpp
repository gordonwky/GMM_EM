#include <Eigen/Dense>
#include <iostream>
// #include <Eigen/Dense>
#include "GMM.h"
#include <cmath>
#include <random>
#include <string>
#include <vector>

// #include "KMeans.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

GMM::GMM() {
  // default constructor
  this->numComponents = 2;
  this->max_iter = 10;
  this->tol = 1e-2;
  this->numData = 0;
  // this->mean_vector = std::vector<VectorXd>(numComponents);
  // this->mixing_coefficients = std::vector<int>(numComponents);
  // this->cov_matrices = std::vector<MatrixXd>(numComponents);
}

GMM::GMM(int numComponents, int max_iter, double tol, std::string init_method) {
  // constructor
  this->numComponents = numComponents;
  this->max_iter = max_iter;
  this->tol = tol;
  this->numData = 0;
  this->init_method = init_method;
  // this->mean_vector = std::vector<VectorXd>(numComponents);
  // this->mixing_coefficients = std::vector<double>(numComponents);
  // this->cov_matrices = std::vector<MatrixXd>(numComponents);
}

void GMM::initiate(const std::vector<VectorXd> &data) {

  if (data.size() == 0) {
    throw std::runtime_error("Dataset is empty to initiate!");
  }
  if (numComponents >= data.size()) {
    throw std::runtime_error("Number of components is greater than or equal "
                             "to the number of data points!");
  }
  this->numData = data.size();
  this->dimData = data[0].size();

  this->responbilities = MatrixXd::Zero(numData, numComponents);
  if (init_method == "random") {
    this->mixing_coefficients = std::vector<double>(
        numComponents, static_cast<double>(1) / numComponents);
    for (int j = 0; j < re_initialize_count; j++) {
      for (int i = 0; i < numComponents; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(0, numData - 1);
        // select a random point as the first centroid
        int random_guess = dist(gen);
        MatrixXd cov = MatrixXd::Identity(dimData, dimData);
        VectorXd mean = data[random_guess];
        cov_matrices.push_back(cov);
        mean_vector.push_back(mean);
      }
      this->normal_distribution_pdf =
          MatrixXd::Zero(numData, numComponents); // normal distribution pdf
      this->likelihood = compute_log_likelihood(data);
      if (std::isnan(likelihood) || std::isinf(likelihood)) {
        std::cout << "Nan / Inf value in the likelihood" << std::endl;
        cov_matrices.clear();
        mean_vector.clear();
        continue;
      } else {
        std::cout << "value in the likelihood " << likelihood << std::endl;
        break;
      }
    }
  } else if (init_method == "kmeans") {
    KMeans kmeans(numComponents, 100);
    kmeans.Initiate(data);
    kmeans.fit(data);
    std::vector<VectorXd> centroids = kmeans.get_mean_vector();
    this->mixing_coefficients = kmeans.get_mixing_coefficients();
    for (int i = 0; i < numComponents; i++) {
      MatrixXd cov = MatrixXd::Identity(dimData, dimData);
      VectorXd mean = centroids[i];
      cov_matrices.push_back(cov);
      mean_vector.push_back(mean);
    }
    this->normal_distribution_pdf =
        MatrixXd::Zero(numData, numComponents); // normal distribution pdf
    this->likelihood = compute_log_likelihood(data);
  } else {
    throw "Invalid initialization method";
  }
}
// computing_normal_distribution_pdf(data);

void GMM::computing_normal_distribution_pdf(const std::vector<VectorXd> &data) {
  // matrix for the normal distribution pdf Nnk

  double sqrt_2pi_pow_n = pow(2 * M_PI, dimData / 2.0);
  // printf("sqrt_2pi_pow_n: %f\n", sqrt_2pi_pow_n);
  for (int j = 0; j < numComponents; j++) {
    MatrixXd inv_cov = cov_matrices[j].inverse();
    VectorXd mean = mean_vector[j];
    double det = cov_matrices[j].determinant();
    for (int i = 0; i < numData; i++) {
      VectorXd x = data[i];
      double coeff = 1.0 / (sqrt_2pi_pow_n * sqrt(det));
      VectorXd diff = x - mean;
      double exponent = -0.5 * (diff.transpose() * inv_cov * diff).trace();
      normal_distribution_pdf(i, j) = coeff * exp(exponent);
    }
  }
}

void GMM::E_step() {
  // update the responsibilities

  // MatrixXd

  for (int i = 0; i < numData; i++) {
    for (int j = 0; j < numComponents; j++) {
      double numerator = mixing_coefficients[j] * normal_distribution_pdf(i, j);
      double denominator = 0.0;

      for (int k = 0; k < numComponents; k++) {
        denominator += mixing_coefficients[k] * normal_distribution_pdf(i, k);
      }

      responbilities(i, j) = numerator / denominator;
    }
  }
}

void GMM::M_step(const std::vector<VectorXd> &data) {
  // update the mean vectors, covariance matrices and mixing coefficients based
  // on responbilities

  for (int i = 0; i < numComponents; i++) {
    VectorXd updated_mean_vector = VectorXd::Zero(dimData);
    MatrixXd updated_cov_matrix = MatrixXd::Zero(dimData, dimData);
    double Nk = 0.0;

    for (int j = 0; j < numData; j++) {
      Nk += responbilities(j, i);
      updated_mean_vector += responbilities(j, i) * data[j];
    }

    updated_mean_vector /= Nk;
    mean_vector[i] = updated_mean_vector;

    for (int j = 0; j < numData; j++) {
      VectorXd diff = data[j] - updated_mean_vector;
      updated_cov_matrix += responbilities(j, i) * diff * diff.transpose();
    }

    updated_cov_matrix /= Nk;
    cov_matrices[i] =
        updated_cov_matrix +
        MatrixXd::Identity(dimData, dimData) * 1e-3; // regularization term
    mixing_coefficients[i] = Nk / numData;
  }
}

void GMM::fit(const std::vector<VectorXd> &data) {
  initiate(data);

  double new_likelihood = 0;
  for (int i = 0; i < max_iter; i++) {
    E_step();
    M_step(data);
    new_likelihood = compute_log_likelihood(data);
    if (std::isnan(new_likelihood) || std::isinf(new_likelihood)) {
      throw std::invalid_argument("The model is not going to converge");
      break;
    }
    if (abs(likelihood - new_likelihood) < tol) {
      break;
    }
    likelihood = new_likelihood;
  }
  fitted = true;
}

std::vector<int> GMM::predict() {

  if (!fitted) {
    throw std::runtime_error("The model has not been fitted yet!");
  }
  std::vector<int> labels(numData);
  for (int i = 0; i < numData; i++) {
    double max_responsibility = 0.0;
    int label = 1;
    for (int j = 0; j < numComponents; j++) {
      if (responbilities(i, j) > max_responsibility) {
        max_responsibility = responbilities(i, j);
        label = j + 1;
      }
    }
    labels[i] = label;
  }
  return labels;
}

double GMM::compute_log_likelihood(const std::vector<VectorXd> &data) {
  double log_likelihood = 0.0;
  computing_normal_distribution_pdf(data);
  for (int i = 0; i < numData; i++) {
    double sum = 0.0;
    for (int j = 0; j < numComponents; j++) {
      sum += mixing_coefficients[j] * normal_distribution_pdf(i, j);
    }
    log_likelihood += log(sum);
  }
  return log_likelihood;
}