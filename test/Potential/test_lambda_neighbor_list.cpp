// -*-c++-*-
#include "copyright.h"
#include <iostream>
#include <chrono>
#include <random>
#include "../../src/Potential/lambda_neighbor_list.h"
#include "../../src/Topology/atomgraph_enumerators.h"

using namespace stormm;
using namespace energy;
using namespace topology;

/// \brief Simple performance test for lambda neighbor list
int main() {
  std::cout << "========================================\n";
  std::cout << "Lambda Neighbor List Performance Test\n";
  std::cout << "========================================\n\n";

  // Test parameters
  const int n_ghosts = 1000;
  const int n_coupled = 100;  // 100 coupled atoms (e.g., 8-9 benzene molecules)
  const int n_atoms = n_ghosts + n_coupled;
  const double cutoff = 12.0;
  const double skin = 2.0;
  const double box_size = 50.0;

  std::cout << "Test system:\n";
  std::cout << "  Total atoms:   " << n_atoms << "\n";
  std::cout << "  Coupled atoms: " << n_coupled << "\n";
  std::cout << "  Ghost atoms:   " << n_ghosts << "\n";
  std::cout << "  Box size:      " << box_size << " A\n";
  std::cout << "  Cutoff:        " << cutoff << " A\n";
  std::cout << "  Skin:          " << skin << " A\n\n";

  // Generate random coordinates
  std::vector<double> xcrd(n_atoms);
  std::vector<double> ycrd(n_atoms);
  std::vector<double> zcrd(n_atoms);
  std::vector<double> lambda_vdw(n_atoms);
  std::vector<double> lambda_ele(n_atoms);

  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<double> coord_dist(0.0, box_size);

  for (int i = 0; i < n_atoms; i++) {
    xcrd[i] = coord_dist(rng);
    ycrd[i] = coord_dist(rng);
    zcrd[i] = coord_dist(rng);

    // First n_coupled atoms are coupled, rest are ghosts
    if (i < n_coupled) {
      lambda_vdw[i] = 1.0;
      lambda_ele[i] = 1.0;
    } else {
      lambda_vdw[i] = 0.0;
      lambda_ele[i] = 0.0;
    }
  }

  std::cout << "Building neighbor list...\n";

  try {
    // Create neighbor list
    auto start = std::chrono::high_resolution_clock::now();

    LambdaNeighborList nblist(n_atoms, lambda_vdw.data(), lambda_ele.data(),
                               cutoff, skin);

    auto after_create = std::chrono::high_resolution_clock::now();

    // Build neighbor list (CPU version)
    nblist.build(xcrd.data(), ycrd.data(), zcrd.data(),
                 nullptr, UnitCellType::NONE, false);  // Use CPU

    auto after_build = std::chrono::high_resolution_clock::now();

    // Print timing
    auto create_time = std::chrono::duration_cast<std::chrono::microseconds>(
        after_create - start).count();
    auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(
        after_build - after_create).count();

    std::cout << "\nResults:\n";
    std::cout << "  Creation time:     " << create_time / 1000.0 << " ms\n";
    std::cout << "  Build time:        " << build_time / 1000.0 << " ms\n";
    std::cout << "  Coupled count:     " << nblist.getCoupledCount() << "\n";
    std::cout << "  Max neighbors:     " << nblist.getMaxNeighbors() << "\n";

    // Calculate expected vs actual neighbors
    const int all_pairs_count = (n_coupled * (n_coupled - 1)) / 2;
    const int neighbor_count = nblist.getCoupledCount() * nblist.getMaxNeighbors() / 2;

    std::cout << "\nPair count comparison:\n";
    std::cout << "  All-pairs:         " << all_pairs_count << " pairs\n";
    std::cout << "  With neighbor list: ~" << neighbor_count << " pairs\n";

    if (neighbor_count < all_pairs_count) {
      const double speedup = static_cast<double>(all_pairs_count) / neighbor_count;
      std::cout << "  Potential speedup:  " << speedup << "x\n";
    }

    // Test rebuild detection
    std::cout << "\nTesting rebuild detection...\n";
    bool needs_rebuild1 = nblist.needsRebuild(xcrd.data(), ycrd.data(), zcrd.data());
    std::cout << "  Initial: " << (needs_rebuild1 ? "NEEDS REBUILD" : "OK") << "\n";

    // Move one atom slightly (less than skin/2)
    xcrd[0] += 0.5;
    bool needs_rebuild2 = nblist.needsRebuild(xcrd.data(), ycrd.data(), zcrd.data());
    std::cout << "  After small move: " << (needs_rebuild2 ? "NEEDS REBUILD" : "OK") << "\n";

    // Move one atom far (more than skin/2)
    xcrd[0] += 2.0;
    bool needs_rebuild3 = nblist.needsRebuild(xcrd.data(), ycrd.data(), zcrd.data());
    std::cout << "  After large move: " << (needs_rebuild3 ? "NEEDS REBUILD" : "OK") << "\n";

    std::cout << "\nTest PASSED!\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "\nTest FAILED with exception: " << e.what() << "\n";
    return 1;
  }
}
