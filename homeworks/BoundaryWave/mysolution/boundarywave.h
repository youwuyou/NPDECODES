#ifndef BOUNDARYWAVE_HPP
#define BOUNDARYWAVE_HPP

/** @file
 * @brief NPDE BoundaryWave
 * @author Erick Schulz
 * @date 24/07/2019
 * @copyright Developed at ETH Zurich
 */

#include <iostream>

// Lehrfem++ includes
#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>
#include <lf/geometry/geometry.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/SparseLU>

namespace BoundaryWave {

// Library functions
lf::assemble::COOMatrix<double> buildM(
    const std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> &fe_space_p);

lf::assemble::COOMatrix<double> buildA(
    const std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> &fe_space_p);

/* SAM_LISTING_BEGIN_7 */
template <typename FUNCTOR_U, typename FUNCTOR_V>
std::pair<Eigen::VectorXd, Eigen::VectorXd> interpolateInitialData(
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space_p,
    FUNCTOR_U &&u0, FUNCTOR_V &&v0) {
    Eigen::VectorXd dof_vector_u0, dof_vector_v0;

    // Generate Lehrfem++ mesh functions out of the functors
    //====================
    // Your code goes here
    // NOTE: u0 and v0 are of type std::function<double(Eigen::Vector2d)>
    //      -> takes vector and returns a double value

    // STEP 1: wrap u0, v0 functors as mesh functions
    lf::mesh::utils::MeshFunctionGlobal mf_u0{u0};
    lf::mesh::utils::MeshFunctionGlobal mf_v0{v0};

    // STEP 2: use NodalProjection to interpolate directly
    // using lf::fe:: prefix
    // dof_vector_u0 = lf::fe::NodalProjection(*fe_space_p, mf_u0);
    // dof_vector_v0 = lf::fe::NodalProjection(*fe_space_p, mf_v0);
    
    // or more concisely
    dof_vector_u0 = NodalProjection(*fe_space_p, mf_u0);
    dof_vector_v0 = NodalProjection(*fe_space_p, mf_v0);

    //====================

    std::pair<Eigen::VectorXd, Eigen::VectorXd> initialData =
    std::make_pair(dof_vector_u0, dof_vector_v0);
    return initialData;
}
/* SAM_LISTING_END_7 */

/* SAM_LISTING_BEGIN_8 */
template <typename FUNCTOR_U, typename FUNCTOR_V>
Eigen::VectorXd solveBoundaryWave(
    const std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> &fe_space_p,
    FUNCTOR_U &&u0, FUNCTOR_V &&v0, double T, unsigned int N) {
    Eigen::VectorXd bdyWaveSol;

    double step_size = T / N;
    // Obtain initial data
    std::pair<Eigen::VectorXd, Eigen::VectorXd> initialData =
        interpolateInitialData<std::function<double(Eigen::Vector2d)>,
                                std::function<double(Eigen::Vector2d)>>(
            fe_space_p, std::move(u0), std::move(v0));
    // Obtain Galerkin matrices
    lf::assemble::COOMatrix<double> M = buildM(fe_space_p);
    lf::assemble::COOMatrix<double> A = buildA(fe_space_p);
    //====================
    // Your code goes here

    // STEP 1: recast data
    // obtain I.C. dof_vector_u0, dof_vector_v0 from the pair
    auto dof_vector_u0 = std::get<0>(initialData);
    auto dof_vector_v0 = std::get<1>(initialData);

    // create Eigen::SparseMatrix
    auto M_sparse = M.makeSparse();
    auto A_sparse = A.makeSparse();
    // Eigen::SparseMatrix<double> M_sparse = M.makeSparse();
    // Eigen::SparseMatrix<double> A_sparse = A.makeSparse();

    // STEP 2: define solver, resolved timestepping using Crank-Nicolson 1-st order scheme
    // vⱼ = (M + 1/4 τ² A)⁻¹ [(*)]
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(M_sparse + 0.25*step_size*step_size*A_sparse);


    // STEP 3: assign initial values to solution vectors
    Eigen::VectorXd u_old = dof_vector_u0;
    Eigen::VectorXd v_old = dof_vector_v0;
    Eigen::VectorXd u;
    Eigen::VectorXd v;

    // fixed no.iterations due to constant timestep size
    for (int i = 0; i < N; ++i){

        // update for the k+1-th iteration
        // (*)= (M-1/4τ²A)vⱼ₋₁ - τA uⱼ₋₁
        v = solver.solve((M_sparse - 0.25*step_size*step_size * A_sparse)*v_old - step_size*A_sparse*u_old);
        u = u_old + 0.5*step_size*(v + v_old);

        // assign old values
        u_old = u;
        v_old = v;
    }

    // return solution at t = T
    bdyWaveSol = u;

    //====================
    return bdyWaveSol;
};
/* SAM_LISTING_END_8 */

}  // namespace BoundaryWave

#endif
