#include <iostream>
#include <eigen-3.4.0/Eigen/Dense>
#include <vector>
#include <fstream>
using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd &read_csv(const std::string &filename)
{

    std::ifstream file(filename);
    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line))
    {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(std::stod(cell));
        }

        data.push_back(row);
    }

    int rows = data.size();
    int cols = data[0].size();
    MatrixXd matrix(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}

MatrixXd &computing_normal_distribution_pdf(const int N, const int K, const std::vector<VectorXd> &data, const std::vector<VectorXd> &mean_vector, const std::vector<MatrixXd> &cov_vector)
{
    // matrix for the normal distribution pdf Nnk
    MatrixXd Nnk(N, K);
    for (size_t i = 0; i < N; i++)
    {
        VectorXd x = data[i];
        for (size_t j = 0; j < K; j++)
        {
            MatrixXd cov = cov_vector[j];
            VectorXd mean = mean_vector[j];
            double det = cov.determinant();
            double coeff = 1.0 / (pow(2 * M_PI, N / 2.0) * sqrt(det));
            VectorXd diff = x - mean;
            MatrixXd inv_cov = cov.inverse();
            double exponent = -0.5 * (diff.transpose() * inv_cov * diff).trace();
            Nnk(i, j) = coeff * exp(exponent);
        }
    }
    return Nnk;
}

double compute_log_likelihood(const std::vector<VectorXd> &data, const std::vector<VectorXd> &mean_vectors, const std::vector<MatrixXd> &cov_matrices, const std::vector<VectorXd> &mixing_coefficients)
{
    int K = mean_vectors.size();
    int N = mean_vectors[0].size();
    int num_samples = data.size();
    double log_likelihood = 0.0;

    for (int i = 0; i < num_samples; i++)
    {
        double sample_likelihood = 0.0;

        for (int j = 0; j < K; j++)
        {
            double component_likelihood = mixing_coefficients[j](0) * computing_normal_distribution_pdf(data[i], mean_vectors[j], cov_matrices[j]);
            sample_likelihood += component_likelihood;
        }

        log_likelihood += log(sample_likelihood);
    }

    return log_likelihood;
}

MatrixXd &computing_responbilities_E_step()
{
    MatrixXd responbilities;

    for (size_t i = 0; i < K; i++)
    {

        for (size_t j = 0; j < N; j++)
        {
        }
    }
    return responbilities;
}

void Update_M_step(const std::vector<VectorXd> &mean_vectors, const std::vector<MatrixXd> &cov_matrices, const std::vector<VectorXd> &mixing_coefficients)
{
    // update the mean vectors, covariance matrices and mixing coefficients based on responbilities
}
void initiate(int K, int N, const std::vector<VectorXd> &data)
{
    std::vector<VectorXd> mean_vectors_vector;
    std::vector<MatrixXd> cov_matrices_vector;
    VectorXd mixing_coefficient = VectorXd::Constant(K, static_cast<double>(1 / K));
    for (int i = 0; i < K; i++)
    {

        MatrixXd cov_matrix = MatrixXd::Identity(N, N);
        VectorXd mean_vector = VectorXd::Random(N);
        cov_matrices_vector.push_back(cov_matrix);
        mean_vectors_vector.push_back(mean_vector);
    }
}

int main()
{

    MatrixXd A = read_csv("iris.csv");
    std::cout << A << std::endl;
    return 0;
}
