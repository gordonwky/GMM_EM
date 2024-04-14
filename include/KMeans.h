#include <iostream>
#include <Eigen/Dense>
// #include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <random>
#include <map>
using Eigen::MatrixXd;
using Eigen::VectorXd;
class KMeans
{
private:
    int numClusters;
    int numIteration;
    std::vector<VectorXd> mean_vector;       // mean
    std::vector<double> mixing_coefficients; // weight
    std::vector<VectorXd> centroids;

public:
    KMeans();
    KMeans(int numClusters, int numIteration);
    void Initiate(const std::vector<VectorXd> &data);
    void fit(const std::vector<VectorXd> &data);

    std::vector<VectorXd> get_mean_vector();
    std::vector<double> get_mixing_coefficients();
};