#!/bin/bash

for build_dir in "models" "results"
do
    if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
    fi
done

#pretrain
python CAND_Multi_Branch_NAS.py --dataset="nn" --max_nodes=20 --init_num=1000 --seed=1
