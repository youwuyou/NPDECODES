/**
 * @ file transformedconslaw.h
 * @ brief NPDE homework about conservation law with non-linear density
 * @ author Ralf Hiptmair, Oliver Rietmann
 * @ date July 2021
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <utility>

namespace TRFCL {

/**
 * @brief Class describing a Cauchy problem with non-linear density
 */
class NonStdCauchyProblemCL {
 public:
  NonStdCauchyProblemCL() = default;
  virtual ~NonStdCauchyProblemCL() = default;
  // Function rho
  double rho(double z) const;
  // Derivative of function rho
  double drho(double z) const;
  // Function g
  double g(double z) const;
  // Derivative of function g
  double dg(double z) const;
  // Finite interval containing "interesting" parts of solution
  std::pair<double, double> domain() const;
  // Final time
  double T() const;
  // Initial data
  double z0(double x) const;
};

/**
 * @brief Newton's method for the inversion of rho
 *
 * Approximately solves rho(z) = u employing Newton's method with initial guess.
 *
 * @tparam RHOFUNCTOR provides rho(z), std::function<double(double)>
 * @tparam DRHOFUNCTOR provides rho'(z), std::function<double(double)>
 * @param u value for which the the inverse of rho should be computed
 * @param z0 initial guess for Newton's method
 * @param rho functor for z -> rho(z)
 * @param rhod functor providing the derivative of rho
 * @param atol absolute tolerance
 * @param rtol relative tolerance

 WRONG IMPLEMENTATION OF TEARMINATION CRITERION.
 */
/* SAM_LISTING_BEGIN_1 */
template <typename RHOFUNCTOR, typename DRHOFUNCTOR>
double rhoInverse(double u, double z0, RHOFUNCTOR &&rho, DRHOFUNCTOR &&drho,
                  double atol = 1.0E-10, double rtol = 1.0E-5) {
  //====================
  // Your code goes here
  double error, norm;

  do{
    // ?? inverse of drho?
    double update = -drho(z0)*rho(z0);

    // update z0, where at the first step it was the initial guess
    z0 = z0 + update;

    // compute error
    error = rho(z0) - u;

    // for relative error comparison
    norm = std::abs(z0);

  }while (error >= atol && error/norm >= rtol);


  //====================
  return z0;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
template <class CAUCHYPROBLEM>
Eigen::VectorXd semiDiscreteRhs(const Eigen::VectorXd &mu,
                                const Eigen::VectorXd &zeta,
                                CAUCHYPROBLEM prb) {
  int N = mu.size();
  Eigen::VectorXd rhs(N);

  //====================
  // Your code goes here


  // mesh width h
  std::pair<double, double> limits = prb.domain();
  double h = (limits.second - limits.first) / N;

  // define the functor Rusanov numerical flux
  // TODO: change flux definition, used just g 
  auto F = [=](double v, double w)->double{ 

    auto rho = [&prb](double v) { return prb.rho(v); };
    auto drho = [&prb](double v) { return prb.drho(v); };


    double rho_inv_v = rhoInverse(v, 1.0, rho, drho);
    double rho_inv_w = rhoInverse(w, 1.0, rho, drho);


    double result = 0.5*(prb.g(rho_inv_v) + prb.g(rho_inv_w)) - 0.5*(w-v) * std::max(std::abs(prb.dg(rho_inv_v)), std::abs(prb.dg(rho_inv_w)));
    return result;
  };

  rhs[0] = -1.0/h*(F(mu[0] , mu[1]) - F(mu[0], mu[0]));

  for (int j = 1; j < N-1; ++j){
    rhs[j] = -1.0/h*(F(mu[j] , mu[j+1]) - F(mu[j-1], mu[j]));
  }
  rhs[N-1] = -1.0/h*(F(mu[N-1] , mu[N-1]) - F(mu[N-2], mu[N-1]));

  //====================

  return rhs;
}

/* SAM_LISTING_END_2 */

/* SAM_LISTING_BEGIN_3 */
template <class CAUCHYPROBLEM, typename RECORDER = std::function<
                                   void(double, const Eigen::VectorXd &)>>
Eigen::VectorXd solveCauchyPrb(
    unsigned int M, unsigned int N, CAUCHYPROBLEM prb,
    RECORDER &&rec = [](double /*time*/, const Eigen::VectorXd &
                        /*zstate*/) -> void {}) {
  // Get inital data for zeta
  std::pair<double, double> limits = prb.domain();
  Eigen::VectorXd x =
      Eigen::VectorXd::LinSpaced(N, limits.first, limits.second);
  auto z0 = [&prb](double y) { return prb.z0(y); };
  Eigen::VectorXd zeta = x.unaryExpr(z0);

  // Compute time step
  double T = prb.T();
  double dt = T / M;

  // Wrap the involved functions for simpler use below
  auto rho = [&prb](double v) { return prb.rho(v); };
  auto drho = [&prb](double v) { return prb.drho(v); };
  auto r = [rho, drho](double u, double z) {
    return rhoInverse(u, z, rho, drho);
  };

  // Get inital data for mu and get the time grid
  Eigen::VectorXd mu = zeta.unaryExpr(rho);
  Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(M + 1, 0.0, T);

  // record solution at initial time
  rec(t[0], zeta);

  for (int k = 0; k < M; ++k) {
    //====================
    // Your code goes here

    Eigen::VectorXd rhs = semiDiscreteRhs(mu, zeta, prb);

    // FIXME: not sure how to use this rhs vector

    double k1 = f(t, u + dt);
    double k2 = f(t+1/3*dt, u + dt*1/3*k1);
    double k3 = f(t+2/3*dt, u + dt*2/3*k2);

    zeta = zeta + dt*(0.25*k1 + 3/4*k2);

    //====================

    // record solution after current time step
    rec(t[k + 1], zeta);
  }

  return zeta;
}
/* SAM_LISTING_END_3 */

}  // namespace TRFCL
