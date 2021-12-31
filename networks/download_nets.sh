#!/bin/bash
for i in  $(seq 1 1 134)
do
    curl http://sli.ics.uci.edu/uai-data/BN/BN_$i.uai > BN_$i.uai
    echo $i
done
