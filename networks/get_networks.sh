#!/bin/bash
for i in  $(seq 1 1 134)
do
    curl http://sli.ics.uci.edu/uai-data/BN/BN_$i.uai > BN_$i.uai
    echo " - File BN_{$i}.uai downloaded." 
done


for file in alarm carpo hailfinder insurance
do
    curl https://www.cs.huji.ac.il/w~galel/Repository/Datasets/$file/$file.bif
    echo " - File BN_{$i}.uai downloaded." 
done

for file in pigs water munin link mildew diabetes barley
do
    curl https://www.cs.huji.ac.il/w~galel/Repository/Datasets/$file/${file^}.bif
    gzip -d ${file^}.bif.gz
    echo " - File {$file^}.bif downloaded." 
done


