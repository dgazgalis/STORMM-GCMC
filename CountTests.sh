#!/bin/bash

if [ -e test.stats.nofile ] ; then
  rm test.stats.nofile
fi
for FI in `ls ${STORMM_BUILD}/test` ; do
  ${STORMM_BUILD}/test/${FI} | grep "Success." >> test.stats.nofile  
done
cat test.stats.nofile | awk '{ print $2 }'
