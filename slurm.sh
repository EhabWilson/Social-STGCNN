# base command
srun -J v0 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash train.sh > srun.log 2>&1

# test
python ./scripts/train-v0.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth --tag  eth --use_lrschd --num_epochs 250 --gpu_num 2

srun -J v0 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 python ./scripts/train-v0.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth --tag  eth --use_lrschd --num_epochs 250 --model social_stgcnn1

# baseline
srun -J v0 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn > srun-baseline.log 2>&1

# v0 atten multiply
srun -J v0 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn0 > srun.log 2>&1

# v1 atten add
srun -J v1 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn1 > srun.log 2>&1

# v2 atten multi without softmax
srun -J v2 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v2.sh > srun.log 2>&1

# v3 atten multiply without sqrt n
srun -J v3 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v3.sh > srun.log 2>&1

# v4 atten mul + linear
srun -J v4 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v4.sh > srun.log 2>&1

# v5 atten add without softmax
srun -J v5 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn5 > srun.log 2>&1

# v6 linear ffn
srun -J v6 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn6 > srun.log 2>&1

# v7 add dropout
srun -J v7 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn7 > srun.log 2>&1

# v8 add ln
srun -J v8 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn8 > srun.log 2>&1

# v9 v1 head
srun -J v9 -N 1 -c 10 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn9 > srun.log 2>&1

# v9 atten add without softmax
NUM=9
srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} > srun-v${NUM}.log 2>&1

## test
NUM=9
srun -J v${NUM}-test -N 1 -p RTX2080Ti --gres gpu:1 --priority 9999999 python ./testmodel.py --log_dir ./tests/v${NUM}  --model social_stgcnn${NUM} > srun-v${NUM}.log 2>&1 
CUDA_VISIBLE_DEVICES=3 python ./testmodel.py --log_dir ./tests/baseline  --model social_stgcnn