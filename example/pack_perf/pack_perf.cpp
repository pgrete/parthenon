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

// Self Include
#include "pack_perf.hpp"

// Standard Includes
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Parthenon Includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <mesh/mesh_pack.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

namespace parthenon {

Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  // only have one package for this app, but will typically have more things added to
  packages["pack_perf"] = pack_perf::Initialize(pin.get());
  return packages;
}

} // namespace parthenon

namespace pack_perf {

void SimpleFluxDivergence(std::shared_ptr<Container<Real>> &in,
                          std::shared_ptr<Container<Real>> &dudt_cont) {
  MeshBlock *pmb = in->pmy_block;

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(interior);

  auto vin = in->PackVariablesAndFluxes({Metadata::Independent});
  auto dudt = dudt_cont->PackVariables({Metadata::Independent});

  auto &coords = pmb->coords;
  int ndim = pmb->pmy_mesh->ndim;
  pmb->par_for(
      "flux divergence", 0, vin.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
        dudt(l, k, j, i) = 0.0;
        dudt(l, k, j, i) +=
            (coords.Area(X1DIR, k, j, i + 1) * vin.flux(X1DIR, l, k, j, i + 1) -
             coords.Area(X1DIR, k, j, i) * vin.flux(X1DIR, l, k, j, i));
        if (ndim >= 2) {
          dudt(l, k, j, i) +=
              (coords.Area(X2DIR, k, j + 1, i) * vin.flux(X2DIR, l, k, j + 1, i) -
               coords.Area(X2DIR, k, j, i) * vin.flux(X2DIR, l, k, j, i));
        }
        if (ndim == 3) {
          dudt(l, k, j, i) +=
              (coords.Area(X3DIR, k + 1, j, i) * vin.flux(X3DIR, l, k + 1, j, i) -
               coords.Area(X3DIR, k, j, i) * vin.flux(X3DIR, l, k, j, i));
        }
        dudt(l, k, j, i) /= -coords.Volume(k, j, i);
      });

}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto package = std::make_shared<StateDescriptor>("pack_perf");
  Params &params = package->AllParams();

  int num_it = pin->GetOrAddInteger("pack", "num_it", 2);
  params.Add("num_it", num_it);

  std::string field_name("myvar");
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  package->AddField(field_name, m);

  return package;
}

Real ComputeAreaOnMesh(parthenon::Mesh *pmesh) {
  auto pack = parthenon::PackVariablesOnMesh(pmesh, "base",
                                             std::vector<std::string>{"in_or_out"});
  IndexRange ib = pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pack.cellbounds.GetBoundsK(IndexDomain::interior);

  Real area = 0.0;
  using policy = Kokkos::MDRangePolicy<Kokkos::Rank<5>>;
  Kokkos::parallel_reduce(
      "pack_perf compute area",
      policy(parthenon::DevExecSpace(), {0, 0, kb.s, jb.s, ib.s},
             {pack.GetDim(5), pack.GetDim(4), kb.e + 1, jb.e + 1, ib.e + 1},
             {1, 1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(int b, int v, int k, int j, int i, Real &larea) {
        larea += pack(b, v, k, j, i) * pack.coords(b).Area(parthenon::X3DIR, k, j, i);
      },
      area);
  // These params are mesh wide. Doesn't matter which meshblock I pull it from.
  const auto &radius = pmesh->pblock->packages["pack_perf"]->Param<Real>("radius");
  return area / (radius * radius);
}

} // namespace pack_perf
