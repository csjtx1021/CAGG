
#best arch:
#cifar10:
#[cifar10-valid] Ours : best arch is [|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|], test_acc is [[94.56, 94.36, 94.2]]
#[cifar10-valid] REA : best arch is [Structure(4 nodes with |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|)], test_acc is [[94.26, 94.68, 94.18]]
#[cifar10-valid] Random : best arch is [Structure(4 nodes with |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|)], test_acc is [[94.34, 94.47, 94.28]]
#[cifar10-valid] REINFORCE : best arch is [Structure(4 nodes with |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|)], test_acc is [[94.26, 94.68, 94.18]]

#cifar100:
#[cifar100] Ours : best arch is [|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|], test_acc is [[73.35, 73.28, 73.88]]
#[cifar100] REA : best arch is [Structure(4 nodes with |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|)], test_acc is [[73.65, 73.03, 72.92]]
#[cifar100] Random : best arch is [Structure(4 nodes with |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|)], test_acc is [[73.35, 73.28, 73.88]]
#[cifar100] REINFORCE : best arch is [Structure(4 nodes with |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|)], test_acc is [[72.99, 72.9, 73.06]]

#ImageNet16-120:
#[ImageNet16-120] Ours : best arch is [|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|], test_acc is [[46.266666666666666, 46.45000001017252, 46.99999998982747]]
#[ImageNet16-120] REA : best arch is [Structure(4 nodes with |nor_conv_1x1~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|)], test_acc is [[47.083333353678384, 46.45000002034505, 47.000000020345055]]
#[ImageNet16-120] Random : best arch is [Structure(4 nodes with |nor_conv_1x1~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|)], test_acc is [[47.083333353678384, 46.45000002034505, 47.000000020345055]]
#[ImageNet16-120] REINFORCE : best arch is [Structure(4 nodes with |nor_conv_1x1~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|)], test_acc is [[46.49999998982747]]




method="Ours" #"REINFORCE" "Random" #"REA"

python test-NAS-Bench-201.py -f "bestarch-$method-cifar10" -s "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|"

python test-NAS-Bench-201.py -f "bestarch-$method-cifar100" -s "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|"

python test-NAS-Bench-201.py -f "bestarch-$method-ImageNet16-120" -s "|nor_conv_3x3~0|+|nor_conv_1x1~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_1x1~2|"





