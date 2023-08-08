/**
 * @file XXX.cc
 * @brief NPDE homework XXX code
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#include "brachistochrone.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace Brachistochrone {

/* SAM_LISTING_BEGIN_0 */
double L2norm(const Eigen::Matrix<double, 2, Eigen::Dynamic> &knots) {
  double norm = 0.;
  // Definition quadrature points (2-point Gauss)
  const double rh = .5 + .5 / std::sqrt(3.);
  const double lh = .5 - .5 / std::sqrt(3.);
  double h = 1. / static_cast<double>(knots.cols() - 1);
  // Loop over all segments of the curve
  for (int i = 0; i < knots.cols() - 1; ++i) {
    // Evaluate the y-component of uh on both quadrature points
    const Eigen::Vector2d uhl = (1 - lh) * knots.col(i) + lh * knots.col(i + 1);
    const Eigen::Vector2d uhr = (1 - rh) * knots.col(i) + rh * knots.col(i + 1);
    // Compute the contribution to the L2 norm of that current segment
    norm += h / 2. * (uhl.squaredNorm() + uhr.squaredNorm());
  }
  return std::sqrt(norm);
}
/* SAM_LISTING_END_0 */

/* SAM_LISTING_BEGIN_8 */
double traveltime(const Eigen::Matrix<double, 2, Eigen::Dynamic> &knots) {
  double out = 0.;
  // Definition quadrature points (2-point Gauss)
  const double rh = .5 + .5 / std::sqrt(3.);
  const double lh = .5 - .5 / std::sqrt(3.);
  const double h = 1. / (knots.cols() - 1);
  // Loop over all segments of the curve
  for (int i = 0; i < knots.cols() - 1; ++i) {
    // Evaluate the y-component of uh on both quadrature points
    const double uh2l = (1 - lh) * knots(1, i) + lh * knots(1, i + 1);
    const double uh2r = (1 - rh) * knots(1, i) + rh * knots(1, i + 1);
    // Compute the length associated to this segment of the curve
    out += h * (knots.col(i) - knots.col(i + 1)).norm() * .5 *
           (1. / std::sqrt(-uh2l) + 1. / std::sqrt(-uh2r));
  }
  return out;
}
/* SAM_LISTING_END_8 */

/* Functions meant for debugging

   These functions provide the "diffusion coefficient" and right-hand side
   source function for the half cycloid solution. Using them, results in a
   linear variational problem, whose solution is still the half cycloid
 */

Eigen::VectorXd coeff_sigma_dbg(
    const Eigen::Matrix<double, 2, Eigen::Dynamic> &knots) {
  int M = knots.cols() - 1;
  Eigen::VectorXd sigma_vec(M);
  // Definition of locations quadrature points (2-point Gauss rule)
  const double rh = .5 + .5 / std::sqrt(3.);
  const double lh = .5 - .5 / std::sqrt(3.);
  const double h = 1. / M;
  // Exact coefficient for half cycloid case
  auto sigma = [](double xi) -> double {
    xi = 0.5 * xi + 0.5;
    return (2.0 / (std::sqrt(2) * M_PI * (1.0 - std::cos(M_PI * xi))));
  };
  for (int i = 0; i < M; ++i) {
    const double xi_minus = h * (i + lh);
    const double xi_plus = h * (i + rh);
    sigma_vec[i] = 0.5 * (sigma(xi_minus) + sigma(xi_plus));
  }
  return sigma_vec;
}

Eigen::Matrix<double, Eigen::Dynamic, 2> sourcefn2_dbg(
    const Eigen::Matrix<double, 2, Eigen::Dynamic> &knots) {
  const std::size_t M = knots.cols() - 1;
  Eigen::Matrix<double, Eigen::Dynamic, 2> f_mat(M, 2);
  // Definition of locations quadrature points (2-point Gauss rule)
  const double rh = .5 + .5 / std::sqrt(3.);
  const double lh = .5 - .5 / std::sqrt(3.);
  const double h = 1. / M;
  // Exact source function for half cycloid case
  auto f = [](double xi) -> double {
    xi = 0.5 * xi + 0.5;
    return (0.5 * M_PI / (std::sqrt(2) * (std::cos(M_PI * xi) - 1.0)));
  };

  for (int i = 0; i < M; ++i) {
    const double xi_minus = h * (i + lh);
    const double xi_plus = h * (i + rh);
    f_mat(i, 0) = f(xi_minus);
    f_mat(i, 1) = f(xi_plus);
  }
  return f_mat;
}

/* SAM_LISTING_BEGIN_1 */
Eigen::VectorXd coeff_sigma(
    const Eigen::Matrix<double, 2, Eigen::Dynamic> &knots) {
  int M = knots.cols() - 1;
  Eigen::VectorXd sigma_vec(M);
  // ===================================================================
  // Your code here

  // define sigma function
  auto sigma = [=](const double v2, const double dv_norm) -> double {
    return 1.0 / (std::sqrt(-v2) * dv_norm);
  };

  // for quadrature nodes
  const double rh = 0.5 + 0.5 /std::sqrt(3.0);
  const double lh = 0.5 - 0.5 /std::sqrt(3.0);
  const double h  = 1.0/static_cast<double>(M);


  for (int j = 0; j < M; ++j) {
    // compute norm of the derivative u'
    const Eigen::VectorXd dv = (knots.col(j + 1) - knots.col(j))/ h;
    const double norm = dv.norm();

    // evaluate quadrature nodes
    const double uh2l  = (1-lh) * knots(1,j) + lh * knots(1, j+1);
    const double uh2r  = (1-rh) * knots(1,j) + rh * knots(1, j+1);

    const double a = sigma(uh2l, norm);
    const double b = sigma(uh2r, norm);

    // compute sigma
    sigma_vec(j) = 0.5 * (a + b);
  }

  // ===================================================================
  return sigma_vec;
}
/* SAM_LISTING_END_1 */


/* SAM_LISTING_BEGIN_2 */
Eigen::Matrix<double, Eigen::Dynamic, 2> sourcefn2(
    const Eigen::Matrix<double, 2, Eigen::Dynamic> &knots) {
  const std::size_t M = knots.cols() - 1;
  Eigen::Matrix<double, Eigen::Dynamic, 2> f_vec(M, 2);
  // ===================================================================
  // Your code here

  auto f = [=](const double dv_norm, const double u2)-> double{ return dv_norm / 2.0 / std::sqrt(-u2)/u2; };

  const double lh  = 0.5 - 0.5/std::sqrt(3.0);
  const double rh = 0.5 + 0.5/std::sqrt(3.0);
  const double h = 1.0/static_cast<double>(M);

  for (int i = 0; i < M; ++i){

    // compute the derivative
    Eigen::VectorXd dv = (knots.col(i) - knots.col(i+1)) / h;
    const double norm = dv.norm();

    // compute quadrature nodes
    const double uh2l   =  (1-lh) * knots(1,i) + lh * knots(1,i+1);
    const double uh2r   =  (1-rh) * knots(1,i) + rh * knots(1,i+1);

    // store values for f_vec
    f_vec(i,0) = f(norm, uh2l);
    f_vec(i,1) = f(norm, uh2r);

  }

  // ===================================================================
  return f_vec;
}
/* SAM_LISTING_END_2 */

/* SAM_LISTING_BEGIN_3 */
Eigen::SparseMatrix<double> matR(
    const Eigen::Matrix<double, 2, Eigen::Dynamic> &knots) {
  const Eigen::Index M = knots.cols() - 1;
  // Reserve enough space for the tridiagonal matrix
  Eigen::SparseMatrix<double> R(M - 1, M - 1);

  R.reserve(Eigen::RowVectorXi::Constant(M - 1, 3));
  // ===================================================================
  // Your code here
  // ===================================================================

  // Clean the CRS format
  R.makeCompressed();
  return R;
}
/* SAM_LISTING_END_3 */

/* SAM_LISTING_BEGIN_4 */
Eigen::VectorXd compute_rhs(
    const Eigen::Matrix<double, 2, Eigen::Dynamic> &knots, Eigen::Vector2d a,
    Eigen::Vector2d b) {
  Eigen::Index M = knots.cols() - 1;
  double h = 1. / static_cast<double>(M);
  // Compute the "coefficients" for all cells
  // A bit wasteful, because we only need sigma[0] and sigma[M-1]
  const Eigen::VectorXd sigma = coeff_sigma(knots);
  const Eigen::Matrix<double, Eigen::Dynamic, 2> f = sourcefn2(knots);
  // Right-hand side vector
  Eigen::VectorXd rhs(2 * (M - 1));
  // Compute head part half of rhs: $\vec{\varphibf}_1$
  rhs.setZero();
  // ===================================================================
  // Your code here
  // ===================================================================
  return rhs;
}
/* SAM_LISTING_END_4 */

/* SAM_LISTING_BEGIN_5 */
void tabiterr(unsigned int M, double rtol, double atol, unsigned int maxit) {
  // Start of the (half) cyloid curve (xi = 0.5)
  Eigen::Vector2d a({0.5 * M_PI - 1.0, -1.0});
  // Initialize b to agree with endpoint of cycloid curve
  Eigen::Vector2d b({M_PI, -2.});
  // ===================================================================
  // Your code here
  // ===================================================================
}
/* SAM_LISTING_END_5 */

/* SAM_LISTING_BEGIN_6 */
void brachistochrone_cvg(std::string filename) {
  // Set minimal number of cells in the mesh
  int M = 8;
  // Start on the cyloid curve (xi = 0.5)
  Eigen::Vector2d a({0.5 * M_PI - 1.0, -1.0});
  // Set b to the endpoint of the cycloid curve
  Eigen::Vector2d b({M_PI, -2.});
  // Initialize parameters for brachistochrone function
  double atol = 1e-6;
  double rtol = 1e-4;
  double itmax = 100;
  std::vector<double> L2errors;
  // ===================================================================
  // Your code here
  // ===================================================================

  // Print a table with results.
  std::cout << "| k   | ||u_h-I_M u_h||_{L2} |\n";
  std::cout << "-----------------------------|\n";
  for (int i = 0; i < L2errors.size() - 1; ++i) {
    std::cout << "| " << std::setw(3) << 8 * std::pow(2, i) << " | "
              << std::setw(20) << std::setprecision(5) << std::scientific
              << L2errors.at(i) << " |\n";
  }
}
/* SAM_LISTING_END_6 */

void iteration_test(std::string filename) {
  const int M = 16;
  const double atol = 1.0E-6;
  const double rtol = 1.0E-4;
  const int maxit = 100;
  // Half cycloid curve as solution
  Eigen::Vector2d a({0.5 * M_PI - 1.0, -1.0});
  Eigen::Vector2d b({M_PI, -2.});
  // Carry out the fixed-point iteration
  std::vector<Eigen::Matrix<double, 2, Eigen::Dynamic>> iterates;
  Eigen::Matrix<double, 2, Eigen::Dynamic> mu = brachistochrone(
      M, a, b, atol, rtol, maxit,
      [&iterates](const Eigen::Matrix<double, 2, Eigen::Dynamic> &mu) -> void {
        iterates.push_back(mu);
      });
  // Write iterates to file
  std::ofstream outfile(filename);
  outfile << "mu_it = [";
  if (outfile.good()) {
    const unsigned int n_it = iterates.size();
    for (int i = 0; i < n_it; ++i) {
      assert(iterates[i].cols() == M + 1);
      for (int k = 0; k <= M; ++k) outfile << iterates[i](0, k) << " ";
      outfile << ";" << std::endl;
      for (int k = 0; k <= M; ++k) outfile << iterates[i](1, k) << " ";
      outfile << ";" << std::endl;
    }
    outfile << " ]" << std::endl;
  }
}

}  // namespace Brachistochrone
