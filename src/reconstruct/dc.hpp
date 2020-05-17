//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef RECONSTRUCT_DC_HPP_
#define RECONSTRUCT_DC_HPP_
//! \file dc.hpp
//  \brief implements donor cell reconstruction

#include "Kokkos_Macros.hpp"
#include "kokkos_abstraction.hpp"
#include "reconstruct/reconstruction.hpp"

using parthenon::ScratchPad2D;

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX1()
//  \brief reconstruct L/R surfaces of the i-th cells

KOKKOS_FORCEINLINE_FUNCTION
void DonorCellX1(parthenon::team_mbr_t const &member, const int k, const int j,
                 const int il, const int iu, const ParArray4D<Real> &q,
                 ScratchPad2D<Real> &ql, ScratchPad2D<Real> &qr) {
  const int nu = q.extent(4) - 1;

  // compute L/R states for each variable
  for (int n = 0; n <= nu; ++n) {
    parthenon::par_for_inner(
        member, il, iu, [&](const int i) { ql(n, i + 1) = qr(n, i) = q(n, k, j, i); });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX2()
//  \brief
KOKKOS_FORCEINLINE_FUNCTION
void DonorCellX2(parthenon::team_mbr_t const &member, const int k, const int j,
                 const int il, const int iu, const ParArray4D<Real> &q,
                 ScratchPad2D<Real> &ql, ScratchPad2D<Real> &qr) {
  const int nu = q.extent(4) - 1;
  // compute L/R states for each variable
  for (int n = 0; n <= nu; ++n) {
    parthenon::par_for_inner(member, il, iu,
                             [&](const int i) { ql(n, i) = qr(n, i) = q(n, k, j, i); });
  }
  return;
}
} // namespace parthenon

#endif // RECONSTRUCT_DC_HPP_