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

void SimpleFluxDivergenceOnMesh(parthenon::Mesh *pmesh, const int num_it) {
  auto vinpack = parthenon::PackVariablesAndFluxesOnMesh(
      pmesh, "base", std::vector<std::string>{"myvar"},
      std::vector<std::string>{"myvar"});
  auto dudtpack =
      parthenon::PackVariablesOnMesh(pmesh, "dUdt", std::vector<std::string>{"myvar"});

  IndexRange ib = vinpack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = vinpack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = vinpack.cellbounds.GetBoundsK(IndexDomain::interior);

  int ndim = pmesh->ndim;
  const int Nl = vinpack.GetDim(4);
  const int Nk = kb.e - kb.s + 1;
  const int Nj = jb.e - jb.s + 1;
  const int Ni = ib.e - ib.s + 1;
  const int NjNi = Nj * Ni;
  const int NkNjNi = Nk * NjNi;
  const int NlNkNjNi = Nl * NkNjNi;
  const int num_blocks = vinpack.GetDim(5);
  const int Nb = num_blocks;
  const int NbNlNkNjNi = Nb * NlNkNjNi;

  std::cout << "Going over " << num_blocks << " blocks." << std::endl;

  Kokkos::Timer timer;

  for (auto ii = 0; ii < num_it; ii++) {
    timer.reset();

    Kokkos::parallel_for(
        "flux divergence",
        parthenon::team_policy(parthenon::DevExecSpace(), num_blocks, Kokkos::AUTO),
        KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
          const int b = team_member.league_rank();
          const auto coords = vinpack.coords(b);
          auto dudt = dudtpack(b);
          auto vin = vinpack(b);

          Kokkos::parallel_for(
              Kokkos::TeamVectorRange<>(team_member, NlNkNjNi), [&](const int idx) {
                const int l = idx / NkNjNi;
                int k = (idx - l * NkNjNi) / NjNi;
                int j = (idx - l * NkNjNi - k * NjNi) / Ni;
                const int i = (idx - l * NkNjNi - k * NjNi - j * Ni) + ib.s;
                j += jb.s;
                k += kb.s;

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
        });
    Kokkos::fence();
    std::cout << "MB " << Ni << " Iteration " << ii << " using "
              << "SimpleTP"
              << " took " << timer.seconds() << std::endl;
  }
  for (auto ii = 0; ii < num_it; ii++) {
    timer.reset();

    Kokkos::parallel_for(
        "flux divergence", Kokkos::RangePolicy<>(0, NbNlNkNjNi),
        KOKKOS_LAMBDA(const int &idx) {
          const int b = idx / NlNkNjNi;
          const int l = (idx - b * NlNkNjNi) / NkNjNi;
          int k = (idx - b * NlNkNjNi - l * NkNjNi) / NjNi;
          int j = (idx - b * NlNkNjNi - l * NkNjNi - k * NjNi) / Ni;
          const int i = idx - b * NlNkNjNi - l * NkNjNi - k * NjNi - j * Ni + ib.s;
          k += kb.s;
          j += jb.s;
          const auto coords = vinpack.coords(b);
          auto dudt = dudtpack(b);
          auto vin = vinpack(b);

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
    Kokkos::fence();
    std::cout << "MB " << Ni << " Iteration " << ii << " using "
              << "flatpack"
              << " took " << timer.seconds() << std::endl;
  }
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
