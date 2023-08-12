/**
 * @ file
 * @ brief NPDE homework on hierarchical error estimation
 * @ author Ralf Hiptmair
 * @ date July 2021
 * @ copyright Developed at SAM, ETH Zurich
 */

#include "hierarchicalerrorestimator.h"

namespace HEST {
/* SAM_LISTING_BEGIN_3 */
Eigen::VectorXd trfLinToQuad(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO1<double>> fes_lin_p,
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fes_quad_p,
    const Eigen::VectorXd &mu) {
  using gdof_idx_t = lf::assemble::gdof_idx_t;
  // Obtain local-to-global index mappings
  const lf::assemble::DofHandler &dh_lin{fes_lin_p->LocGlobMap()};
  const lf::assemble::DofHandler &dh_quad{fes_quad_p->LocGlobMap()};
  LF_ASSERT_MSG(dh_lin.Mesh() == dh_quad.Mesh(),
                "DofHandlers must be based on the same mesh");
  LF_ASSERT_MSG(dh_lin.NumDofs() == mu.size(), "Vector length mismath");
  // Underlying mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p{dh_lin.Mesh()};
  const lf::mesh::Mesh &mesh{*mesh_p};
  LF_ASSERT_MSG(
      (dh_lin.NumDofs() == mesh.NumEntities(2)) &&
          (dh_quad.NumDofs() == mesh.NumEntities(2) + mesh.NumEntities(1)),
      "#DOFs do not match #entities");
  // The coefficients of a FE function in the quadratic Lagrangian FE space with
  // respect to the nodal basis are simply its values at the nodes and the
  // midpoints of the edges.
  Eigen::VectorXd nu(dh_quad.NumDofs());
//====================
// Your code goes here

  // // go through all nodes of mesh
  int codim_vertices = 2;   // d=2
  for (auto *node: mesh.Entities(codim_vertices)){

    // obtain indices of GSF on nodes for linear FE space
    nonstd::span<const gdof_idx_t> linear_dof_idx{dh_lin.GlobalDofIndices(*node)};

    // obtain indices of GSF on nodes for quadratic FE space
    nonstd::span<const gdof_idx_t> quad_dof_idx{dh_quad.GlobalDofIndices(*node)};

    // direct assignment!
    nu[quad_dof_idx[0]] = mu[linear_dof_idx[0]];
  }

  // // go through all edges of mesh, interpolate from linear to quad space
  int codim_edges    = 1;   // d=2
  for (auto* edge: mesh.Entities(codim_edges)){

    // obtain indices of GSF on edges for linear FE space
    nonstd::span<const gdof_idx_t> linear_dof_idx{dh_lin.GlobalDofIndices(*edge)};

    // obtain indices of GSF on edges for quadratic FE space

    // NOTE: use InteriorGlobalDofIndices to obtain indices for nodes that are contained in this edge!
    // nonstd::span<const gdof_idx_t> quad_dof_idx{dh_quad.GlobalDofIndices(*edge)};  <- WRONG!
    nonstd::span<const gdof_idx_t> quad_dof_idx{dh_quad.InteriorGlobalDofIndices(*edge)};

    // for every edge, use values on endpoints and assign values to quad space directly
    // use 1/2(vh(p1) + vh(p2)) for edge GSF value interpolation
    nu[quad_dof_idx[0]] = 0.5*(mu[linear_dof_idx[0]] + mu[linear_dof_idx[1]]);

  }

//====================
  return nu;
}
/* SAM_LISTING_END_3 */


std::tuple<double, double, double> solveAndEstimate(
    std::shared_ptr<const lf::mesh::Mesh> mesh_p) {
  // Note: the mesh must cover the unit square for this test setting !
  // Create finite element space for p.w. linear Lagrangian FE
  std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> lfe_space_p =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);
  std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO2<double>> quad_space_p =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_p);

  // Define homogeneous Dirichlet boundary value problem
  // Diffusion coefficient function
  std::function<double(Eigen::Vector2d)> alpha =
      [](Eigen::Vector2d x) -> double { return (1.0 /* + x.squaredNorm() */); };
  // Manufactured exact solution
  auto u_exact = [](Eigen::Vector2d x) -> double {
    return (x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]));
  };
  auto grad_u = [](Eigen::Vector2d x) -> Eigen::Vector2d {
    return ((Eigen::Vector2d() << (1 - 2 * x[0]) * x[1] * (1 - x[1]),
             x[0] * (1 - x[0]) * (1 - 2 * x[1]))
                .finished());
  };
  // Right-hand side matching exact solution
  auto f = [](Eigen::Vector2d x) -> double {
    return 2 * (x[1] * (1 - x[1]) + x[0] * (1 - x[0]));
  };

  // Lambdas have to be wrapped into a mesh function for error computation
  lf::mesh::utils::MeshFunctionGlobal mf_u{u_exact};
  lf::mesh::utils::MeshFunctionGlobal mf_grad_u{grad_u};
  lf::mesh::utils::MeshFunctionGlobal mf_alpha{alpha};
  lf::mesh::utils::MeshFunctionGlobal mf_f{f};

  // Compute basis expansion coefficient vector of finite-element solution
  const Eigen::VectorXd mu{solveBVPWithLinFE(mf_alpha, mf_f, lfe_space_p)};

  // Compute error norms
  // create MeshFunctions representing solution / gradient of LFE solution
  const lf::fe::MeshFunctionFE mf_sol(lfe_space_p, mu);
  const lf::fe::MeshFunctionGradFE mf_grad_sol(lfe_space_p, mu);
  // compute errors with 3rd order quadrature rules, which is sufficient for
  // piecewise linear finite elements
  double L2err =  // NOLINT
      std::sqrt(lf::fe::IntegrateMeshFunction(
          *mesh_p, lf::mesh::utils::squaredNorm(mf_sol - mf_u), 2));
  double H1serr = std::sqrt(lf::fe::IntegrateMeshFunction(  // NOLINT
      *mesh_p, lf::mesh::utils::squaredNorm(mf_grad_sol - mf_grad_u), 2));

  std::cout << "Mesh (" << mesh_p->NumEntities(0) << " cells, "
            << mesh_p->NumEntities(1) << " edges, " << mesh_p->NumEntities(2)
            << " nodes): L2err = " << L2err << ", H1serr = " << H1serr
            << std::endl;

  // Evaluate a-posteriori error estimator
  const Eigen::VectorXd nu =
      compHierSurplusSolution(mf_alpha, mf_f, lfe_space_p, quad_space_p, mu);
  // Compute H1-seminorm of solution in hierarchical surplus space
  const lf::fe::MeshFunctionGradFE mf_grad_hps(quad_space_p, nu);
  double hier_surplus_norm = std::sqrt(lf::fe::IntegrateMeshFunction(  // NOLINT
      *mesh_p, lf::mesh::utils::squaredNorm(mf_grad_hps), 4));
  std::cout << "Estimated error = " << hier_surplus_norm << std::endl;

  return {L2err, H1serr, hier_surplus_norm};
}  // end solveAndEstimate

}  // namespace HEST
