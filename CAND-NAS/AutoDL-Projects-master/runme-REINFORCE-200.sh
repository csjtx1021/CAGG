for seed in 1 2 3 4 5 #1 2 3 4 5 #1 2 3 4 5
do

     bash ./scripts-search/algos/REINFORCE-200.sh cifar10 0.001 $seed

done

for seed in 1 2 3 4 5
do

    bash ./scripts-search/algos/REINFORCE-200.sh cifar100 0.001 $seed

done

for seed in 1 2 3 4 5
do
    bash ./scripts-search/algos/REINFORCE-200.sh ImageNet16-120 0.001 $seed
done
