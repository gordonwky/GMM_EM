#include <Eigen/Dense>
#include <iostream>
// #include <Eigen/Dense>
#include "KMeans.h"
#include <fstream>
#include <map>
#include <omp.h>
#include <random>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

KMeans::KMeans() : numClusters(2), numIteration(200) {
  // default constructor
  // set the number of clusters to 2
  // set the number of iterations to 200
  // set the initial centroids to empty
  // set the mixing coefficients to empty
  // set the mean vector to empty with size 2
  centroids.reserve(2);
  mean_vector.reserve(2);
  mixing_coefficients.reserve(2);
}

KMeans::KMeans(int numClusters, int numIteration)
    : numClusters(numClusters), numIteration(numIteration) {
  // constructor with parameters
  // set the number of clusters to the given number of clusters
  // set the number of iterations to the given number of iterations
  // set the initial centroids to empty
  // set the mixing coefficients to empty
  // set the mean vector to empty with size numClusters
  centroids.reserve(numClusters);
  mean_vector.reserve(numClusters);
  mixing_coefficients.reserve(numClusters);
}

void KMeans::Initiate(const std::vector<VectorXd> &data) {
  // Initialize the labels with K-means++ algorithm
  // Implement the K-means algorithm here
  // Return the initial centroids
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, data.size() - 1);
  // select a random point as the first centroid

  centroids[0] = data[dist(gen)];

  std::vector<int> labels(data.size());

  for (int i = 1; i < numClusters; i++) {
    std::vector<double> distances;
    for (size_t j = 0; j < data.size(); j++) {
      double init_distance = 0;
      for (size_t k = 0; k < centroids.size(); k++) {
        init_distance += (data[j] - centroids[k]).norm();
      }
      distances.push_back(init_distance);
    }
    int index =
        std::distance(distances.begin(),
                      std::max_element(distances.begin(), distances.end()));
    try {
      if (index < 0 || index >= data.size()) {
        throw std::out_of_range("Index out of range");
      } else if (std::find(centroids.begin(), centroids.end(), data[index]) !=
                 centroids.end()) {
        throw std::invalid_argument("Centroid already exists");
      } else {
        centroids.push_back(data[index]);
      }
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
    }
  }
}

void KMeans::fit(const std::vector<VectorXd> &data) {
  Initiate(data); // Only run once, no threading.

  std::vector<std::vector<int>> assignments(
      numClusters); // thread-safe writing required

  for (int iter = 0; iter < numIteration; ++iter) {
    for (auto &cluster : assignments)
      cluster.clear();

    std::vector<int> point_to_cluster(data.size());

// Parallel assignment step
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
      int best = 0;
      double best_dist = (data[i] - centroids[0]).squaredNorm();
      for (int j = 1; j < numClusters; ++j) {
        double dist = (data[i] - centroids[j]).squaredNorm();
        if (dist < best_dist) {
          best = j;
          best_dist = dist;
        }
      }
      point_to_cluster[i] = best;
    }

    // Group assignments (single-threaded, safe)
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
      assignments[point_to_cluster[i]].push_back(i);
    }

    // Update centroids (can also be parallelized, but less gain)
    for (int k = 0; k < numClusters; ++k) {
      if (!assignments[k].empty()) {
        VectorXd new_centroid = VectorXd::Zero(data[0].size());
        for (int idx : assignments[k]) {
          new_centroid += data[idx];
        }
        centroids[k] = new_centroid / assignments[k].size();
      }
    }
  }

  // Final mixing coefficients
  mixing_coefficients.resize(numClusters);
  for (int k = 0; k < numClusters; ++k) {
    mixing_coefficients[k] =
        static_cast<double>(assignments[k].size()) / data.size();
  }

  mean_vector = centroids;
}

std::vector<VectorXd> KMeans::get_mean_vector() { return this->mean_vector; }

std::vector<double> KMeans::get_mixing_coefficients() {
  return this->mixing_coefficients;
}