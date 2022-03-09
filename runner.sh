#!/bin/bash
name="UNIT"
module load anaconda3/2020.11-nold

source activate /private/home/rbalestriero/.conda/envs/ffcv

for i in {1..3}
do
    for lr in "0.01"
    do
        for epoch in 300 500
        do
            for wd in "5e-4" "0.0"
            do
                for ls in "0.0" "0.1"
                do
                    for momentum in "0" "0.9"
                    do
                        for scheduler in step triangle
                        do
                            while [ $(squeue -u $USER | grep $name | wc -l) -gt 50 ]
                            do
                                echo 'Waiting...'
                                sleep 1m
                            done
                            command="python train_cifar.py --config-file cifar_config.yaml --training.lr $lr --training.epochs $epoch --training.weight_decay $wd --training.momentum $momentum --training.label_smoothing $ls --training.scheduler $scheduler"
                            echo $command
                            sleep 0.3s
                            sbatch --job-name=$name --partition=scavenge --nodes=1 --time=2000 --cpus-per-task=10 --ntasks-per-node=1 --gpus-per-task=1 --wrap="$command"
                        done
                    done
                done
            done
        done
    done
done


