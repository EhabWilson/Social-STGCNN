# !/bin/bash
MODEL=$1
NAME=$2
LR=$3
# SEED=$RANDOM
SEED=8177
envs=(
    "eth"
    "hotel"
    "zara1"
    "univ"
    "zara2"
)
id=0
for env in ${envs[*]}; do
    python ./scripts/train-v3.py --model ${MODEL} --name ${NAME} --ilr ${LR} --n_stgcnn 1 --n_txpcnn 5  --dataset ${env} --tag ${env} --use_lrschd --num_epochs 250 --gpu_num ${id}  &
    echo "${env} Launched."
    id=$(($id+1))
done
echo ${SEED}
wait
