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
//=======================================================================================
#ifndef MESH_MESH_PACK_HPP_
#define MESH_MESH_PACK_HPP_

#include <array>
#include <utility>

#include "coordinates/coordinates.hpp"
#include "interface/container.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp" // TODO(JMM): Replace with forward declaration?

namespace parthenon {

// a separate dims array removes a branch case in `GetDim`
// TODO(JMM): Using one IndexShape because its the same for all
// meshblocks. This needs careful thought before sticking with it.
template <typename T>
class MeshPack {
 public:
  MeshPack() = default;
  MeshPack(const ParArray1D<T> view, const IndexShape shape,
           const ParArray1D<Coordinates_t> coordinates, const std::array<int, 5> dims)
      : v_(view), cellbounds(shape), coords(coordinates), dims_(dims) {}
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block) const { return v_(block); }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block, const int n) const { return v_(block)(n); }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int block, const int n, const int k, const int j,
                   const int i) const {
    return v_(block)(n)(k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetDim(const int i) const {
    assert(i > 0 && i < 6);
    return dims_[i - 1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNdim() const { return v_(0).GetNdim(); }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetSparse(const int n) const { return v_(0).GetSparse(n); }

  // TODO(JMM): Also include mesh domain object?
  IndexShape cellbounds;
  ParArray1D<Coordinates_t> coords;

 private:
  ParArray1D<T> v_;
  std::array<int, 5> dims_;
};

template <typename T>
using ViewOfPacks = ParArray1D<VariablePack<T>>;
template <typename T>
using ViewOfFluxPacks = ParArray1D<VariableFluxPack<T>>;
template <typename T>
using MeshVariablePack = MeshPack<VariablePack<T>>;
template <typename T>
using MeshVariableFluxPack = MeshPack<VariableFluxPack<T>>;

// TODO(JMM): Should this be cached?
namespace mesh_pack_impl {
template <typename T, typename F>
auto PackMesh(const Mesh *pmesh, F &packing_function) {
  int nblocks = pmesh->GetNumMeshBlocksThisRank();
  ParArray1D<T> packs("MakeMeshVariablePack::view", nblocks);
  auto packs_host = Kokkos::create_mirror_view(packs);
  ParArray1D<Coordinates_t> coords("MakeMeshPackVariable::coords", nblocks);
  auto coords_host = Kokkos::create_mirror_view(coords);

  // TODO(JMM): Update to Andrew's C++ std::list when available
  MeshBlock *pmb = pmesh->pblock;
  int b = 0;
  while (pmb != nullptr) {
    coords_host(b) = pmb->coords;
    packs_host(b) = packing_function(pmb);
    pmb = pmb->next;
    b++;
  }

  std::array<int, 5> dims;
  for (int i = 0; i < 4; i++) {
    dims[i] = packs_host(0).GetDim(i + 1);
  }
  dims[4] = nblocks;

  Kokkos::deep_copy(packs, packs_host);
  Kokkos::deep_copy(coords, coords_host);

  return MeshPack<T>(packs, pmesh->pblock->cellbounds, coords, dims);
}
} // namespace mesh_pack_impl

// Uses Real only because meshblock only owns real containers
template <typename... Args>
auto PackVariablesOnMesh(const Mesh *pmesh, const std::string &container_name,
                         Args &&... args) {
  using namespace mesh_pack_impl;
  auto pack_function = [&](MeshBlock *pmb) {
    auto container = pmb->real_containers.Get(container_name);
    return container->PackVariables(std::forward<Args>(args)...);
  };
  return PackMesh<VariablePack<Real>>(pmesh, pack_function);
}
template <typename... Args>
auto PackVariablesAndFluxesOnMesh(const Mesh *pmesh, const std::string &container_name,
                                  Args &&... args) {
  using namespace mesh_pack_impl;
  auto pack_function = [&](MeshBlock *pmb) {
    auto container = pmb->real_containers.Get(container_name);
    return container->PackVariablesAndFluxes(std::forward<Args>(args)...);
  };
  return PackMesh<VariableFluxPack<Real>>(pmesh, pack_function);
}

} // namespace parthenon

#endif // MESH_MESH_PACK_HPP_