Application: Neural Architecture Search (NAS), including cell-based NAS and  multi-branch NAS

This code is implemented according to paper "Cost-Aware Graph Generation: A Deep Bayesian Optimization Approach". Cost-Aware Graph Generation (CAGG) can generate optimal graphs at as low cost s
as possible.  We apply it to two challenging real-world problems, i.e., molecular discovery and neural architecture
search, to rigorously evaluate its effectiveness and applicability.

If you want to run this code, you should ensure that you have installed the following tools and packages:

    NASBench201 by ``pip install nas-bench-201"
    torch
    seaborn

Download data:

You can download the NASBench201 dataset for cell-based NAS from [its project](https://github.com/D-X-Y/NAS-Bench-201) (NOTE: we use the ``NAS-Bench-201-v1_1-096897.pth" in our paper).
    The datasets used in Multi-Branch NAS can be downloaded from [IndoorLoc.p](https://drive.google.com/open?id=1FEXzEvyRGNFm9GP-v4hEnJrJtTCLVJo1) and [SliceLocalization.p](https://drive.google.com/open?id=1T_FXqwIWt-AxZBiwCmWSIIJs4oTZqqWw)
    
After downloading these datasets, you should move them into folder ``data/".
    
After installing all dependency packages and preparing the datasets, you can run this code with the default setting.

Or you can see the help message by running as:
    
    $$ python CAGG_Cell_based_NAS.py -h
    
    or
    
    $$ python CAGG_Multi_Branch_NAS.py -h

Run this code:

(Before running the CAGG, you should make the dir first by hand or running following codes in shell:

    for build_dir in "models" "results"
    do
        if [ ! -d "$build_dir" ]; then
            mkdir $build_dir
        fi
    done

(1) Pretrain the generation model in a VAE fashion by running as:

    $$ python CAGG_Multi_Branch_NAS.py --dataset="nn" --max_nodes=20 --init_num=1000 --seed=1
    
or, run an example as

    $$ bash pretrain.sh
    
(NOTE: In NAS problems, only Multi-Branch NAS needs the pretrain.)
    
(2) Run CAGG code: 

For cell-based NAS:

    $$ python CAGG_Cell_based_NAS.py --dataset="NASBench201" --max_nodes=4 --max_iter=390 --init_num=10 --image_data="cifar100" --store_file_name="results/observations.csv" --seed=1

DATA: When you want choose image data, you can use the [--image_data] option.

For multi-branch NAS:
    
    $$ python CAGG_Multi_Branch_NAS.py --dataset="nn" --max_nodes=20 --max_iter=60 --init_num=10 --store_file_name="results/observations.csv" --pretrain_name="models/orig_rand1000_epoch70" --seed=1
    
DATA: When you want to choose different MLP data, you can open one of lines 15 and 16 in ``objective_func.py'' as follows

    mlp_dataset_name="data/IndoorLoc.p" #input_dim=521
    #mlp_dataset_name="data/SliceLocalization.p" #input_dim=385

(please see these examples in ``run_cand.sh")



