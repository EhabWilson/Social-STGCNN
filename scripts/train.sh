# !/bin/bash

envs=(
    "eth"
    "hotel"
    "univ"
    "zara1"
    "zara2"
)

for env in ${envs[*]}; do
    python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset ${env} --tag ${env} --use_lrschd --num_epochs 250  &
    echo "${env} Launched."
done
wait