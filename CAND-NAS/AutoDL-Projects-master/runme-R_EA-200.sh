for seed in 1 2 3 4 5 #1 2 3 4 5
do

    bash ./scripts-search/algos/R-EA-200.sh cifar10 3 $seed

done

for seed in 1 2 3 4 5
do

    bash ./scripts-search/algos/R-EA-200.sh cifar100 3 $seed

done

for seed in 1 2 3 4 5
do
    bash ./scripts-search/algos/R-EA-200.sh ImageNet16-120 3 $seed
done
