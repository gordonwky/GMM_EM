#ifndef MIXTURE_HPP
#define MIXTURE_HPP
#include <Eigen/Dense>
class Mixture {
public:
  Mixture();
  Mixture(int numComponents, int max_iter, double tol, std::string init_method);
  void initiate_parameters();
  virtual void initiate(const std::vector<Eigen::VectorXd> &data);
  virtual void
  computing_normal_distribution_pdf(const std::vector<Eigen::VectorXd> &data);
  virtual void E_step();
  virtual void M_step(const std::vector<Eigen::VectorXd> &data);
  virtual void fit(const std::vector<Eigen::VectorXd> &data);
  virtual double
  compute_log_likelihood(const std::vector<Eigen::VectorXd> &data);
  virtual std::vector<int> predict();
  virtual ~Mixture() {}

private:
  int re_initialize_count = 50;
  int numComponents;
  int max_iter;
  double tol;
  int numData;
  int dimData;
  bool fitted = false;
  std::string init_method = "random";
  double likelihood = 1e30;
};

#endif