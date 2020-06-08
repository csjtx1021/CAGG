for seed in 1 2 3 4 5 #1 2 3 4 5 #1 2 3 4 5
do

    bash ./scripts-search/algos/Random-200.sh cifar10 $seed

done

for seed in 1 2 3 4 5
do

    bash ./scripts-search/algos/Random-200.sh cifar100 $seed

done

for seed in 1 2 3 4 5
do
    bash ./scripts-search/algos/Random-200.sh ImageNet16-120 $seed
done
