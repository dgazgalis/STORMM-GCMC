#!/bin/bash

cat > md.in << EOF
&files
  -sys { -p ${STORMM_SOURCE}/test/Namelists/topol/gly_arg.top
         -c ${STORMM_SOURCE}/test/Namelists/coord/gly_arg.inpcrd
         -label GlyArg -n 16 }
  -sys { -p ${STORMM_SOURCE}/test/Namelists/topol/gly_arg.top
         -c ${STORMM_SOURCE}/test/Namelists/coord/gly_arg.inpcrd
         -label GlyArg_II -n 50 }
  -sys { -p ${STORMM_SOURCE}/test/Namelists/topol/gly_arg.top
         -c ${STORMM_SOURCE}/test/Namelists/coord/gly_arg.inpcrd
         -label GlyArg_III -n 50 }
  -o mdn.out
&end

&minimize
  cdcyc 20,  ncyc 40,  maxcyc 60,
  ntpr 1,
&end

&dynamics
  nstlim = 2000,  ntpr = 25,  ntwx = 0, dt = 1.0,
  ntt = 0,
  rigid_geom on,
  temperature = { tempi 100.0, temp0 300.0, -label GlyArg },
  temperature = { tempi 100.0, temp0 400.0, -label GlyArg_II },
  temperature = { tempi 300.0, temp0 200.0, -label GlyArg_III },
  tevo_start = 250, tevo_end = 750,
  tcache_depth 1,
&end

&solvent
  igb = 8,
&end

&report
  syntax = Matlab,
  energy total,
&end
EOF

${STORMM_BUILD}/apps/Dyna/dynamics.stormm -O -i md.in -except warn
