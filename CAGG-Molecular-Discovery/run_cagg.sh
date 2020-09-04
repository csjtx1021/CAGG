#!/bin/bash

for build_dir in "models" "results"
do
    if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
    fi
done

seed=1
#do CAGG if you want to choose the initialized networks from dataset randomly
#python CAGG.py --dataset="qm9" --max_nodes=9 --max_iter=500 --init_num=50 --property_name="5*QED-SA" --store_file_name="results/observations.csv" --pretrain_name="models/orig_rand1000" --seed=$seed
#Or
#do CAGG when there are exist initialized networks in file, you should provide the file name, such as:
python CAGG.py --dataset="qm9" --max_nodes=9 --max_iter=500 --init_num=50 --property_name="5*QED-SA" --store_file_name="results/observations.csv" --pretrain_name="models/orig_rand1000" --exist_init_nets="data/QM9_init/init_smiles_r$seed.txt" --seed=$seed
