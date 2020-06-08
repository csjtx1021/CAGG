#!/bin/bash

for build_dir in "models" "results"
do
    if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
    fi
done

#pretrain
python CAND.py --dataset="qm9" --max_nodes=9 --init_num=1000 --seed=1
