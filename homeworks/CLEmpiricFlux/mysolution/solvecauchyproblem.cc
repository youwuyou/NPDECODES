/**
 * @file solvecauchyproblem.cc
 * @brief NPDE exam problem summer 2019 "CLEmpiricFlux" code
 * @author Oliver Rietmann
 * @date 19.07.2019
 * @copyright Developed at ETH Zurich
 */

#include "solvecauchyproblem.h"

#include <Eigen/Core>
#include <cmath>

#include "uniformcubicspline.h"

namespace CLEmpiricFlux {

/* SAM_LISTING_BEGIN_1 */
Eigen::Vector2d findSupport(const UniformCubicSpline &f,
                            Eigen::Vector2d initsupp, double t) {
  Eigen::Vector2d result;
  //====================
  // Your code goes here

  // STEP 1: find out the speeds of propagation
  Eigen::Vector2d speed(2);
  speed << f.derivative(-1.0), f.derivative(1.0);

  // STEP 2: add on the distance to the initial support
  // xᵢ = xᵢ₋₁ + vt
  result = initsupp + speed*t;

  //====================
  return result;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
template <typename FUNCTOR>
Eigen::VectorXd semiDiscreteRhs(const Eigen::VectorXd &mu0, double h,
                                FUNCTOR &&numFlux) {
  int m = mu0.size();
  Eigen::VectorXd mu1(m);
  //====================
  // Your code goes here
  //
  mu1[0] = -1.0/h*(numFlux(mu0[0], mu0[1]) - numFlux(mu0[0], mu0[0]));

  for (int j = 1; j < m-1; ++j){
    mu1[j] = -1.0/h*(numFlux(mu0[j], mu0[j+1]) - numFlux(mu0[j-1], mu0[j]));
  }

  mu1[m-1] = -1.0/h*(numFlux(mu0[m-1], mu0[m-1]) - numFlux(mu0[m-2], mu0[m-1]));

  //====================
  return mu1;
}
/* SAM_LISTING_END_2 */

/* SAM_LISTING_BEGIN_3 */
template <typename FUNCTOR>
Eigen::VectorXd RalstonODESolver(FUNCTOR &&rhs, Eigen::VectorXd mu0, double tau,
                                 int n) {
  //====================
  // Your code goes here

  // timestepping with constant timestep size tau
  for (int i = 0; i < n; ++ i){

    // compute k1, k2
    Eigen::VectorXd k1 = rhs(mu0);
    Eigen::VectorXd k2 = rhs(mu0 + tau*(2.0/3.0)*k1);

    // update 
    mu0    = mu0 + tau*(0.25*k1 + 3.0/4.0*k2);
  }
  //====================
  return mu0;
}
/* SAM_LISTING_END_3 */

/* SAM_LISTING_BEGIN_4 */
Eigen::VectorXd solveCauchyProblem(const UniformCubicSpline &f,
                                   const Eigen::VectorXd &mu0, double h,
                                   double T) {
  Eigen::VectorXd muT(mu0.size());
  //====================
  // Your code goes here
  // STEP 1: determine tau using CFL condition (11.4.2.12)
  double s_min = f.derivative(-1.0);
  double s_max = f.derivative(1.0);
  double tau = h/std::max(std::abs(s_min),std::abs(s_max));
  int    n   = T/tau;

  // STEP 2: obtain semi-discretized RHS 
  GodunovFlux numFlux(f);
  auto rhs = [=](Eigen::VectorXd u)->Eigen::VectorXd{ return semiDiscreteRhs(u, h, numFlux);};

  // STEP 3: timestepping loop embedded in the solver
  muT = RalstonODESolver(rhs, mu0, tau, n);
  //====================
  return muT;
}
/* SAM_LISTING_END_4 */

}  // namespace CLEmpiricFlux
