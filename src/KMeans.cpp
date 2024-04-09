#include <iostream>
#include <Eigen/Dense>
// #include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <random>
#include <map>

#include "KMeans.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

KMeans::KMeans()
{
    this->numClusters = 2;
    this->numIteration = 200;
}

KMeans::KMeans(int numClusters, int numIteration)
{
    this->numClusters = numClusters;
    this->numIteration = numIteration;
}

void KMeans::Initiate(const std::vector<VectorXd> &data)
{
    // Initialize the labels with K-means++ algorithm
    // Implement the K-means algorithm here
    // Return the initial centroids
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, numClusters);
    // select a random point as the first centroid
    int first_centroid = dist(gen);
    VectorXd centroid = data[first_centroid];

    this->centroids.push_back(centroid);
    std::vector<int> labels(data.size());

    for (int i = 1; i < numClusters; i++)
    {
        std::vector<double> distances;
        for (int j = 0; j < data.size(); j++)
        {
            double init_distance = 0;
            for (int k = 0; k < this->centroids.size(); k++)
            {
                init_distance += (data[j] - this->centroids[k]).norm();
            }
            distances.push_back(init_distance);
        }
        int index = std::distance(distances.begin(), std::max_element(distances.begin(), distances.end()));
        try
        {
            if (index < 0 || index >= data.size())
            {
                throw std::out_of_range("Index out of range");
            }
            else if (std::find(this->centroids.begin(), this->centroids.end(), data[index]) != this->centroids.end())
            {
                throw std::invalid_argument("Centroid already exists");
            }
            else
            {
                this->centroids.push_back(data[index]);
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
}

void KMeans::fit(const std::vector<VectorXd> &data)
{
    // Implement the K-means algorithm here
    // Return the labels for each data point
    Initiate(data);
    // std::vector<int> labels(data.size());
    std::map<int, std::vector<VectorXd>> clusters;
    for (int i = 0; i < numIteration; i++)
    {
        // train the KMeans model
        for (int i = 0; i < numClusters; i++)
        {
            clusters[i] = std::vector<VectorXd>();
        }
        for (int j = 0; j < data.size(); j++)

        {
            double min_distance = 1e30;
            int index = 0;
            for (int k = 0; k < numClusters; k++)
            {
                // assign the data point to the closest centroid
                if ((data[j] - centroids[k]).norm() < min_distance)
                {
                    index = k;
                    min_distance = (data[j] - centroids[k]).norm();
                }
            }
            clusters[index].push_back(data[j]);
        }
        // update the centroids
        for (int i = 0; i < numClusters; i++)
        {
            VectorXd centroid_new = VectorXd::Zero(data[0].size());
            for (int j = 0; j < clusters[i].size(); j++)
            {
                centroid_new += clusters[i][j];
            }
            centroid_new /= clusters[i].size();
            centroids[i] = centroid_new;
        }
    }
    this->mixing_coefficients = std::vector<double>(numClusters);
    for (int i = 0; i < numClusters; i++)
    {
        // cout number in each cluster
        // std::cout << "Cluster " << i << " size: " << static_cast<double>(clusters[i].size()) / static_cast<double>(data.size()) << std::endl;
        // for (int j = 0; j < clusters[i].size(); j++)
        // {
        //     std::cout << clusters[i][j] << std::endl;
        // }
        mixing_coefficients[i] = static_cast<double>(clusters[i].size()) / static_cast<double>(data.size());
    }
    this->mean_vector = centroids;
}

std::vector<VectorXd> KMeans::get_mean_vector()
{
    return this->mean_vector;
}

std::vector<double> KMeans::get_mixing_coefficients()
{
    return this->mixing_coefficients;
}