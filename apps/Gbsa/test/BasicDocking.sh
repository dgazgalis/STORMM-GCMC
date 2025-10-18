#!/bin/bash

cat > nrg_est.in << EOF
&files
  -sys { -p ${STORMM_SOURCE}/test/Topology/symmetry_C1.top
         -c symmetry_C1.crd -label ligand }
  -sys { -p ${STORMM_SOURCE}/test/Topology/symmetry_C4.top
         -c symmetry_C4.crd -label ligand }
  -sys { -p ${STORMM_SOURCE}/test/Topology/symmetry_C3.top
         -c symmetry_C3.crd -label receptor }
&end


&mmgbsa
  weights boltzmann,
  depth averages,
  temperature 100.0,
  carveout { cutoff 35.0 },
&end

&solvent
  igb = 8,
&end

&minimize
  cdcyc = 50,  ncyc = 100,  maxcyc = 500,
  clash_r0 = 1.4, clash_vdw_ratio = 0.8,
  ntpr = 10,
&end

&report
  syntax Matlab,
  report_width 99,
&end
EOF

valgrind ${STORMM_BUILD}/apps/Gbsa/mmgbsa.stormm.cuda -O -i nrg_est.in -except WARN

