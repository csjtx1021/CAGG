#!/bin/bash

for build_dir in "models" "results"
do
    if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
    fi
done

seed=1

#For cell-based NAS:
python CAND_Cell_based_NAS.py --dataset="NASBench201" --max_nodes=4 --max_iter=390 --init_num=10 --image_data="cifar100" --store_file_name="results/observations.csv" --seed=$seed

#For multi branch NAS:
#python CAND_Multi_Branch_NAS.py --dataset="nn" --max_nodes=20 --max_iter=60 --init_num=10 --store_file_name="results/observations.csv" --pretrain_name="models/orig_rand1000_epoch70" --seed=$seed



