/** @file
 * @brief NPDE BoundaryWave
 * @author Erick Schulz
 * @date 24/07/2019
 * @copyright Developed at ETH Zurich
 */

#include "boundarywave.h"

namespace BoundaryWave {

/* SAM_LISTING_BEGIN_1 */
lf::assemble::COOMatrix<double> buildM(
  const std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> &fe_space_p) {
  // I. TOOLS AND DATA
  // Pointer to current fe_space and mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p(fe_space_p->Mesh());
  // Obtain local->global index mapping for current finite element space
  const lf::assemble::DofHandler &dofh{fe_space_p->LocGlobMap()};
  // Dimension of finite element space
  const lf::uscalfe::size_type N_dofs(dofh.NumDofs());

  // II : ASSEMBLY
  // Matrix in triplet format holding Galerkin matrix, zero initially.
  lf::assemble::COOMatrix<double> M(N_dofs, N_dofs);
  //====================
  // Your code goes here

  // build matrix M using m(v, w) = ∫ ∂Ω v(x)w(x) dS(x)

  // STEP 1: find out edges on the boundary
  lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{ lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};

  // // create predicate
  auto edge_predicate = [=](const lf::mesh::Entity &edge)->bool { return bd_flags(edge);};

  // STEP 2: initialize object of type MassEdgeMatrixProvider
  // note: for initializing MassEdgeMatrixProvider, we need to have gamma defined using MeshFunction
  // 
  auto gamma = lf::mesh::utils::MeshFunctionConstant<double>(1.0); 

  // alternatively:
  // lf::uscalfe::MassEdgeMatrixProvider<double, decltype(gamma), decltype(edge_predicate)> 
  //     edgemat_builder(fe_space_p, gamma, edge_predicate);
  lf::uscalfe::MassEdgeMatrixProvider
      edgemat_builder(fe_space_p, gamma, edge_predicate);


  // // STEP 3: assemble matrix using the AssembleMatrixLocally function
  // //  edgemat_builder is of the type EntityMatrixProvider
  lf::assemble::AssembleMatrixLocally(1, dofh, dofh, edgemat_builder, M);

  //====================
  return M;
};
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
lf::assemble::COOMatrix<double> buildA(
    const std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> &fe_space_p) {
  // I. TOOLS AND DATA
  // Pointer to current fe_space and mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p(fe_space_p->Mesh());
  // Obtain local->global index mapping for current finite element space
  const lf::assemble::DofHandler &dofh{fe_space_p->LocGlobMap()};
  // Dimension of finite element space
  const lf::uscalfe::size_type N_dofs(dofh.NumDofs());

  // II : ASSEMBLY
  // Matrix in triplet format holding Galerkin matrix, zero initially.
  lf::assemble::COOMatrix<double> A(N_dofs, N_dofs);
  //====================
  // Your code goes here
  // Recall ReactionDiffusionElementMatrixProvider
  // for a(u,v) = ∫ Ω ɑ(x) grad u ·grad v + γ(x) uv dx

  // now we have
  //     a(u,v) = ∫ Ω (1+ ||x||²)grad u ·grad v dx

  // STEP 1: define diffusion coeff function
  auto alpha = [=](Eigen::Vector2d x)->double{ double norm = x.norm(); return (1.0 + norm*norm);};

  lf::mesh::utils::MeshFunctionGlobal mf_alpha{alpha};

  // STEP 2: define reaction coeff function = 0
  lf::mesh::utils::MeshFunctionConstant<double> mf_gamma(0.0);

  // STEP 3: create element matrix provider
  lf::uscalfe::ReactionDiffusionElementMatrixProvider
      mat_provider(fe_space_p, mf_alpha, mf_gamma);
  
  // STEP 4: assemble matrix
  lf::assemble::AssembleMatrixLocally(0 , dofh, dofh, mat_provider, A);

  //====================
  return A;
};

/* SAM_LISTING_END_2 */

}  // namespace BoundaryWave
