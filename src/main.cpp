#include <iostream>
#include <eigen-3.4.0/Eigen/Dense> // #include <Eigen/Dense>
#include <vector>
#include <fstream>
#include "GMM.h"
#include "input.h"
#include "output.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main()
{
    // command to compile
    // g++ -std=c++17 GMM.cpp KMeans.cpp input.cpp output.cpp main.cpp

    // std::vector<VectorXd> data;
    // data.push_back(VectorXd(2));
    // data[0] << 1, 2;
    // data.push_back(VectorXd(2));
    // data[1] << 2, 3;
    // data.push_back(VectorXd(2));
    // data[2] << 3, 4;
    // data.push_back(VectorXd(2));
    // data[3] << 4, 5;
    // data.push_back(VectorXd(2));
    // data[4] << 5, 6;
    // data.push_back(VectorXd(2));
    // data[5] << 10, 11;
    // data.push_back(VectorXd(2));
    // data[6] << 11, 12;

    std::vector<VectorXd> data = csv_to_data("/Users/kimyingwong/GMM_EM/dataset/iris.csv");
    // std::ifstream file("/Users/kimyingwong/GMM_EM/dataset/iris.csv");
    // std::string line;

    // while (std::getline(file, line))
    // {
    //     std::vector<double> row;
    //     std::stringstream ss(line);
    //     std::string cell;

    //     while (std::getline(ss, cell, ','))
    //     {
    //         row.push_back(std::stod(cell));
    //     }

    //     VectorXd vec(row.size());
    //     for (int i = 0; i < row.size(); i++)
    //     {
    //         vec(i) = row[i];
    //     }
    //     data.push_back(vec);
    // }

    std::cout << "Data: " << std::endl;
    for (const auto &vec : data)
    {
        std::cout << vec.transpose() << std::endl;
    }

    GMM gmm(3, 100, 1e-2, "kmeans");
    // // gmm.initiate(data);
    gmm.fit(data);
    // // std::cout << "Responbilities: " << std::endl;
    // // std::cout
    // //     << gmm.responbilities << std::endl;
    std::vector<int> label = gmm.predict();
    // std::cout << "Labels: " << std::endl;
    // for (const auto &l : label)
    // {
    //     std::cout << l << std::endl;
    // }
    writeLabelsToCSV(label, "/Users/kimyingwong/GMM_EM/dataset/testing.csv");
    // std::vector<VectorXd> mean_vector = gmm.mean_vector;
    // std::cout << "Mean Vector: " << std::endl;
    // for (const auto &mean : mean_vector)
    // {
    //     std::cout << mean << std::endl;
    // }
    // std::vector<double> mixing_coefficients = gmm.mixing_coefficients;
    // std::cout << "Mixing Coefficients: " << std::endl;
    // for (const auto &coef : mixing_coefficients)
    // {
    //     std::cout << coef << std::endl;
    // }
    // Test case 2
    // std::ofstream outputFile("../dataset/test_lables.csv");
    // if (outputFile.is_open())
    // {
    //     outputFile << "labels"
    //                << ","
    //                << "\n";
    //     for (const auto &l : label)
    //     {
    //         outputFile << l << ","
    //                    << "\n";
    //     }
    //     outputFile.close();
    //     std::cout << "Labels saved to labels.csv" << std::endl;
    // }
    // else
    // {
    //     std::cout << "Unable to open file" << std::endl;
    // }
    return 0;
}