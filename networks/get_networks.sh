#!/bin/bash
for file in alarm carpo hailfinder insurance
do
    curl https://www.cs.huji.ac.il/w~galel/Repository/Datasets/$file/$file.bif
    echo " - File BN_{$i}.uai downloaded." 
done


