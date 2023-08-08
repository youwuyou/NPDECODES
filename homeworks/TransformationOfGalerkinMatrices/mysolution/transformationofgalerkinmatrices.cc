/**
 * @file
 * @brief NPDE homework TransformationOfGalerkinMatrices code
 * @author Erick Schulz
 * @date 01/03/2019
 * @copyright Developed at ETH Zurich
 */

#include "transformationofgalerkinmatrices.h"

#include <cassert>
#include <iostream>

namespace TransformationOfGalerkinMatrices {

/* SAM_LISTING_BEGIN_1 */
std::vector<Eigen::Triplet<double>> transformCOOmatrix(
    const std::vector<Eigen::Triplet<double>> &A) {
  std::vector<Eigen::Triplet<double>> A_t{};  // return value

  // First step: find the size of the matrix by searching the maximal
  // indices. Depends on the assumption that no zero rows/columns occur.
  int rows_max_idx = 0, cols_max_idx = 0;
  for (const Eigen::Triplet<double> &triplet : A) {
    rows_max_idx =
        (triplet.row() > rows_max_idx) ? triplet.row() : rows_max_idx;
    cols_max_idx =
        (triplet.col() > cols_max_idx) ? triplet.col() : cols_max_idx;
  }
  int n_rows = rows_max_idx + 1;
  int n_cols = cols_max_idx + 1;

  // Make sure we deal with a square matrix
  assert(n_rows == n_cols);
  // The matrix size must have even parity
  assert(n_cols % 2 == 0);

  int N = n_cols;      // Size of (square) matrix
  int M = n_cols / 2;  // Half the size
  //====================
  // Your code goes here
  // iterate over triplets of A
  for (auto& it: A){

    const int k = it.row() + 1;
    const int l = it.col() + 1;
    auto val = it.value();


    // std::cout<< "debugging! Row: " << k << "; Col: " << l << " Val: "<< val << std::endl;

    // case distinction
    if (k % 2 == 0 && l % 2 == 0){
      // k, l even
      A_t.emplace_back(k/2 -1,l/2 -1, val);
      A_t.emplace_back(k/2+M -1,l/2+M -1, val);
      A_t.emplace_back(k/2+M -1,l/2 -1, -val);
      A_t.emplace_back(k/2 -1,l/2+M -1, -val);
    }
    else if(k % 2 != 0  && l % 2 != 0){
      // k, l odd
      A_t.emplace_back((k+1)/2 -1,(l+1)/2 -1, val);
      A_t.emplace_back((k+1)/2+M -1,(l+1)/2+M -1, val);
      A_t.emplace_back((k+1)/2 -1,(l+1)/2+M -1, val);
      A_t.emplace_back((k+1)/2+M -1,(l+1)/2 -1, val);
    }

    else if (k % 2 == 0 && l % 2 != 0){
      // k even, l odd
      A_t.emplace_back((k)/2 -1,(l+1)/2 -1, val);
      A_t.emplace_back((k)/2 -1,(l+1)/2+M -1, val);
      A_t.emplace_back((k)/2+M -1,(l+1)/2+M -1, -val);
      A_t.emplace_back((k)/2+M -1,(l+1)/2 -1, -val);
    }

    else{
      // k odd, l even
      A_t.emplace_back((k+1)/2 -1,(l)/2 -1, val);
      A_t.emplace_back((k+1)/2+M -1,(l)/2+M -1, -val);
      A_t.emplace_back((k+1)/2+M -1,(l)/2 -1, val);
      A_t.emplace_back((k+1)/2 -1,(l)/2+M -1, -val);

    }
  }



  //====================
  return A_t;
}
/* SAM_LISTING_END_1 */

}  // namespace TransformationOfGalerkinMatrices
