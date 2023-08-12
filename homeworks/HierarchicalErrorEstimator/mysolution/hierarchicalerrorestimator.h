/**
 * @ file
 * @ brief NPDE homework on hierarchical error estimation
 * @ author Ralf Hiptmair
 * @ date July 2021
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <lf/assemble/assemble.h>
#include <lf/base/base.h>
#include <lf/fe/fe.h>
#include <lf/geometry/geometry.h>
#include <lf/mesh/mesh.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>

namespace HEST {

template <typename MESHFUNCTION_ALPHA, typename MESHFUNCTION_F>
Eigen::VectorXd solveBVPWithLinFE(
    const MESHFUNCTION_ALPHA &mf_alpha, const MESHFUNCTION_F &mf_f,
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fes_lin_p) {
  // For conveneicne we set up references to essential objects for FE
  // discretization in the lowest-order Lagrangian finite element space
  const lf::uscalfe::FeSpaceLagrangeO1<double> &linfespc{*fes_lin_p};
  // The underlying finite-element mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p{linfespc.Mesh()};
  const lf::mesh::Mesh &mesh{*mesh_p};
  // Obtain local->global index mapping for current finite element space
  const lf::assemble::DofHandler &dofh{linfespc.LocGlobMap()};
  // Dimension of linear finite element space, number of unknowns
  const lf::base::size_type N_dofs(dofh.NumDofs());

  // I: Assembly of full Galerkin matrix
  // Object for sparse matrix to be filled by cell-oriented assembly
  lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);
  // Provider object for element matrices for scalar linear second-order pure
  // diffusion operator with variable diffusion coefficient. Uses numerical
  // quadrature of order 3 and, thus, computes exact element matrices for
  // locally constant diffusion coefficient.
  lf::fe::DiffusionElementMatrixProvider<double, MESHFUNCTION_ALPHA>
      elmat_builder(fes_lin_p, mf_alpha);
  // Invoke cell-oriented assembly of the finite-element Galerkin matrix
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, A);

  // II: Assembly of right-hand-side vector
  Eigen::Matrix<double, Eigen::Dynamic, 1> phi(N_dofs);
  phi.setZero();
  // Assemble volume part of right-hand side vector depending on the source
  // function f.
  // Initialize object taking care of local computations on all cells.
  lf::uscalfe::ScalarLoadElementVectorProvider<double, MESHFUNCTION_F>
      elvec_builder(fes_lin_p, mf_f);
  // Invoke assembly on cells (codim == 0)
  AssembleVectorLocally(0, dofh, elvec_builder, phi);

  // III: Enforce zero Dirichlet (essential) boundary conditions
  // Create a predicate selecting nodes on the boundary
  lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{
      lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 2)};
  lf::assemble::FixFlaggedSolutionCompAlt<double>(
      [&bd_flags,
       &dofh](lf::assemble::glb_idx_t dof_idx) -> std::pair<bool, double> {
        const lf::mesh::Entity &node{dofh.Entity(dof_idx)};
        return (bd_flags(node) ? std::make_pair(true, 0.0)
                               : std::make_pair(false, 0.0));
      },
      A, phi);
  // Assembly completed: Convert COO matrix A into CRS format using Eigen's
  // internal conversion routines.
  Eigen::SparseMatrix<double> A_crs = A.makeSparse();

  // Solve linear system using Eigen's sparse direct elimination
  // Examine return status of solver in case the matrix is singular
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_crs);
  LF_VERIFY_MSG(solver.info() == Eigen::Success, "LU decomposition failed");
  Eigen::VectorXd sol_vec = solver.solve(phi);
  LF_VERIFY_MSG(solver.info() == Eigen::Success, "Solving LSE failed");
  return sol_vec;
}

/**
 * @brief This function enforces Dirichlet zero boundary conditions on the
 * Galerkin stiffness and mass matrices
 *
 * This function first annihilates all selected rows and columns of a matrix in
 * triplet format. Then the corresponding diagonal entries are set to 1, thus
 * preserving the invertibilty of the matrix
 *
 * @param selectvals The predicate identifying selecting the rows and columns to
 * be set to zero
 * @param A matrix in LehrFEM++ internal triplet format. Will be modified!
 */
template <typename SCALAR, typename SELECTOR>
void dropMatrixRowsColumns(SELECTOR &&selectvals,
                           lf::assemble::COOMatrix<SCALAR> &A) {
  const lf::assemble::size_type N(A.cols());
  LF_ASSERT_MSG(A.rows() == N, "Matrix must be square!");
  // Set the selected rows and columns to zero
  A.setZero(
      [&selectvals](lf::assemble::gdof_idx_t i, lf::assemble::gdof_idx_t j) {
        return (selectvals(i) || selectvals(j));
      });
  // Set the diagonal entries of zeroed out rows and columns to 1
  for (lf::assemble::gdof_idx_t dofnum = 0; dofnum < N; ++dofnum) {
    const auto selval{selectvals(dofnum)};
    if (selval) {
      A.AddToEntry(dofnum, dofnum, 1.0);
    }
  }
}

Eigen::VectorXd trfLinToQuad(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fes_lin_p,
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fes_quad_p,
    const Eigen::VectorXd &mu);

/* SAM_LISTING_BEGIN_3 */
template <typename MESHFUNCTION_ALPHA, typename MESHFUNCTION_F>
Eigen::VectorXd compHierSurplusSolution(
    const MESHFUNCTION_ALPHA &mf_alpha, const MESHFUNCTION_F &mf_f,
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fes_lin_p,
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fes_quad_p,
    const Eigen::VectorXd &mu) {

  // References to FE space
  const lf::uscalfe::FeSpaceLagrangeO2<double> &quad_space{*fes_quad_p};
  const lf::uscalfe::FeSpaceLagrangeO1<double> &lfe_space{*fes_lin_p};
  // Get references to DofHandlers
  const lf::assemble::DofHandler &dh_quad{quad_space.LocGlobMap()};
  const lf::assemble::DofHandler &dh_lfe{lfe_space.LocGlobMap()};
  LF_ASSERT_MSG(dh_lfe.Mesh() == dh_quad.Mesh(),
                "DofHandlers must be based on the same mesh");
  LF_ASSERT_MSG(dh_lfe.NumDofs() == mu.size(), "Vector length mismath");
  // Underlying mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p{dh_lfe.Mesh()};
  const lf::mesh::Mesh &mesh{*mesh_p};
  LF_ASSERT_MSG(
      (dh_lfe.NumDofs() == mesh.NumEntities(2)) &&
          (dh_quad.NumDofs() == mesh.NumEntities(2) + mesh.NumEntities(1)),
      "#dof mismatch");
  // Dimension of the quadratic finite element space
  const lf::base::size_type N_qdofs(dh_quad.NumDofs());
  // Vector for returning the result
  Eigen::VectorXd quad_surplus(N_qdofs);
//====================
// Your code goes here

  // STEP 1: assemble full Galerkin matrix for quadratic FE space
  lf::assemble::COOMatrix<double> A(N_qdofs, N_qdofs);
  lf::fe::DiffusionElementMatrixProvider elmat_builder(fes_quad_p, mf_alpha);
  lf::assemble::AssembleMatrixLocally(0, dh_quad, dh_quad, elmat_builder, A);

  // STEP 2: assemble of RHS vector in quadratic FE space
  Eigen::VectorXd phi(N_qdofs);
  lf::uscalfe::ScalarLoadElementVectorProvider elvec_builder(fes_quad_p, mf_f);
  lf::assemble::AssembleVectorLocally(0, dh_quad, elvec_builder, phi);

  // STEP 3: compute residual vector
  // given a vector mu in linear FE space, obtain its interpolated representation in quadratic FE space
  auto A_temp = A.makeSparse();
  Eigen::VectorXd mu_quad = trfLinToQuad(fes_lin_p, fes_quad_p, mu);
  // Eigen::VectorXd residual_quad = phi + A.MatVecMult(-1.0, mu_quad);
  Eigen::VectorXd residual_quad = phi - A_temp*mu_quad;

  // STEP 4: compute argumented marix for the surplus space
  lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};

  // iterating over all edges, records index of GFS to be dropped
  std::vector<bool> dropped_ed(N_qdofs, false);
  for (auto* edge: mesh.Entities(1)){
    nonstd::span<const lf::assemble::gdof_idx_t> qf_dof_idx{ dh_quad.GlobalDofIndices(*edge)};

    // NOTE: number in lehrfem from nodes -> edges, first two GSF are of endpoints on an edge
    dropped_ed[qf_dof_idx[0]] = true;
    dropped_ed[qf_dof_idx[1]] = true;

    // drop the edge only if it locates on the boundary! Keep all other basis functions
    if (bd_flags(*edge)){
     dropped_ed[qf_dof_idx[2]] = true;
    }
  }

  // eliminate corresponding entries of the Galerkin matrix
  auto selector = [=](lf::assemble::gdof_idx_t idx)->bool{return dropped_ed[idx];};
  dropMatrixRowsColumns(selector, A);

  // eliminate corresponding entries of the rhs residual vector
  for (int j = 0; j < N_qdofs; ++j){
    if (dropped_ed[j] == 1){
      residual_quad[j] = 0.0;
    }
  }
  
  // STEP 5: solve the LSE
  Eigen::SparseMatrix<double> A_crs = A.makeSparse();
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_crs);

  quad_surplus = solver.solve(residual_quad);


//====================
  return quad_surplus;
}
/* SAM_LISTING_END_3 */

std::tuple<double, double, double> solveAndEstimate(
    std::shared_ptr<const lf::mesh::Mesh> mesh_p);

}  // namespace HEST
