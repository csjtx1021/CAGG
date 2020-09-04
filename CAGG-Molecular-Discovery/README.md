Application: Molecular Discovery, including two molecular properties, i.e., 5*QED-SA and logP-SA

This code is implemented according to paper "Cost-Aware Graph Generation: A Deep Bayesian Optimization Approach". Cost-Aware Graph Generation (CAGG) can generate optimal graphs at as low cost 
as possible.  We apply it to two challenging real-world problems, i.e., molecular discovery and neural architecture
search, to rigorously evaluate its effectiveness and applicability.

If you want to run this code, you should ensure that you have installed the following tools and packages:

    rdkit
    torch
    seaborn

After installing all dependency packages, you can run this code with the default setting.

Or you can see the help message by running as:
    
    $$ python CAGG.py -h

Run this code:

(Before running the CAGG, you should make the dir first by hand or running following codes in shell:

    for build_dir in "models" "results"
    do
        if [ ! -d "$build_dir" ]; then
            mkdir $build_dir
        fi
    done

(1) Pretrain the generation model in a VAE fashion by running as:

    $$ python CAGG.py --dataset="qm9" --max_nodes=9 --init_num=1000 --seed=1
    
or, run an example as

    $$ bash pretrain.sh
    
(2) Run CAGG code: 

If you want to choose the initialized networks from dataset randomly, you can run the following code:
    
    $$ python CAGG.py --dataset="qm9" --max_nodes=9 --max_iter=500 --init_num=50 --property_name="5*QED-SA" --store_file_name="results/observations.csv" --pretrain_name="models/orig_rand1000" --seed=1

or, when there are exist initialized networks in file, you should provide the file name, such as
    
    $$ python CAGG.py --dataset="qm9" --max_nodes=9 --max_iter=500 --init_num=50 --property_name="5*QED-SA" --store_file_name="results/observations.csv" --pretrain_name="models/orig_rand1000" --exist_init_nets="data/QM9_init/init_smiles_r1.txt" --seed=1
    
Property: You can choose different property using ``property_name'' option.

(please see the details and these examples in ``run_cagg.sh")



