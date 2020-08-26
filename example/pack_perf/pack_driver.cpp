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

// Standard Includes
#include <fstream>
#include <string>
#include <vector>

// Parthenon Includes
#include <parthenon/package.hpp>

// Local Includes
#include "pack_driver.hpp"
#include "pack_perf.hpp"

// Preludes
using namespace parthenon::package::prelude;

using pack::PackDriver;

int main(int argc, char *argv[]) {
  ParthenonManager pman;

  auto manager_status = pman.ParthenonInit(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  PackDriver driver(pman.pinput.get(), pman.pmesh.get());

  auto driver_status = driver.Execute();

  // call MPI_Finalize if necessary
  pman.ParthenonFinalize();

  return 0;
}

parthenon::DriverStatus PackDriver::Execute() {
  // this is where the main work is orchestrated
  // No evolution in this driver.  Just calculates something once.
  // For evolution, look at the EvolutionDriver
  PreExecute();

  pouts->MakeOutputs(pmesh, pinput);
  const auto &num_it = pmesh->pblock->packages["pack_perf"]->Param<int>("num_it");

  MeshBlock *pmb = pmesh->pblock;
  while (pmb != nullptr) {
    auto &base = pmb->real_containers.Get();
    pmb->real_containers.Add("dUdt", base);
    pmb = pmb->next;
  }

  Kokkos::Timer timer;

  if (pinput->GetOrAddBoolean("pack", "use_mesh_pack", false)) {
    // Use the mesh pack and do it all in one step
    pack_perf::SimpleFluxDivergenceOnMesh(pmesh, num_it);
  } else {
    auto f = [&](auto loop_pattern, auto name) {
      for (auto i = 0; i < num_it; i++) {
        // not using the TaskList here so this is an idealized situation
        // just measuring raw performance (not taking into account async execution)
        MeshBlock *pmb = pmesh->pblock;
        auto range = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
        timer.reset();
        while (pmb != nullptr) {
          auto &base = pmb->real_containers.Get();
          auto &dUdt = pmb->real_containers.Get("dUdt");
          pack_perf::SimpleFluxDivergence(loop_pattern, base, dUdt);

          pmb = pmb->next;
        }
        Kokkos::fence();
        std::cout << "MB " << range.e - range.s + 1 << " Iteration " << i << " using "
                  << name << " took " << timer.seconds() << std::endl;
      }
    };
    f(parthenon::loop_pattern_flatrange_tag, "flatrange");
    f(parthenon::loop_pattern_mdrange_tag, "mdrange");
    f(parthenon::loop_pattern_tpttr_tag, "TPTTR");
    f(parthenon::loop_pattern_tptvr_tag, "TPTVR");
    f(parthenon::loop_pattern_tpttrtvr_tag, "TPTTRTVR");
  }
  return DriverStatus::complete;
}
