#!/bin/bash

cat > md.in << EOF
&files
  -sys { -p ${STORMM_SOURCE}/test/Topology/tip3p.top
         -c ${STORMM_SOURCE}/test/Trajectory/tip3p.inpcrd
         -label TIP3P -n 1 }
  -sys { -p ${STORMM_SOURCE}/test/Topology/tip4p.top
         -c ${STORMM_SOURCE}/test/Trajectory/tip4p.inpcrd
         -label TIP3P_II -n 1 }
  -sys { -p ${STORMM_SOURCE}/test/Topology/ubiquitin.top
         -c ${STORMM_SOURCE}/test/Trajectory/ubiquitin.inpcrd
         -label UBI -n 1 }
  -o mdn.out
&end

&dynamics
  nstlim = 1000,  ntpr = 1,  ntwx = 0, dt = 1.0,
  ntt = 0,
  rigid_geom on,
  temperature = { tempi 100.0, temp0 300.0, -label TIP3P },
  temperature = { tempi 100.0, temp0 400.0, -label TIP3P_II },
  temperature = { tempi 300.0, temp0 200.0, -label UBI },
  tevo_start = 250, tevo_end = 750,
  tcache_depth 1,
&end

&precision
  nonbonded double,
  valence double,
&end

&report
  syntax = Matlab,
  energy total,
&end
EOF

${STORMM_BUILD}/apps/Dyna/dynamics.stormm -O -i md.in -except warn
