#!/bin/bash
# bash ./scripts-search/algos/REINFORCE.sh 0.001 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset, LR, and seed"
  exit 1
fi
data_dir="../data"
if [ "$data_dir" = "" ]; then
  echo "Must set data_dir envoriment variable for data dir saving"
  exit 1
else
  echo "data_dir : $data_dir"
fi

dataset=$1
LR=$2
seed=$3
channel=16
num_cells=5
max_nodes=4
space=nas-bench-201
benchmark_file=${data_dir}/NAS-Bench-201-v1_0-e61699.pth
#benchmark_file=${TORCH_HOME}/NAS-Bench-201-v1_1-096897.pth

save_dir=./output/search-cell-${space}/REINFORCE-200-${dataset}-${LR}

OMP_NUM_THREADS=4 python ./exps/algos/reinforce-200.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} \
	--search_space_name ${space} \
	--arch_nas_dataset ${benchmark_file} \
	--time_budget 12000 \
	--learning_rate ${LR} --EMA_momentum 0.9 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
