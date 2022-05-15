# !/bin/bash
MODEL=$1
NAME=$2
DROP=$3
INIT=$4
K=$5
# SEED=$RANDOM
SEED=8177
envs=(
    "eth"
    "hotel"
    "univ"
    "zara1"
    "zara2"
)

for env in ${envs[*]}; do
    python ./scripts/train-v0.py --model ${MODEL} --name ${NAME} --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset ${env} --tag ${env} --use_lrschd --num_epochs 250  --drop ${DROP} --init ${INIT} --seed ${SEED} --dict_kernel_size ${K}
    echo "${env} Launched."
done
echo ${SEED}
wait
