NUM=9 
srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 > srun-v${NUM}-0.log 2>&1 &
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 > srun-v${NUM}-0.log 2>&1 &
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.1 > srun-v${NUM}-1.log 2>&1 &
# srun -J v${NUM} -N 1 -p RTX3090 -w node06 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.2 > srun-v${NUM}-2.log 2>&1 &

# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.3 > srun-v${NUM}-3.log 2>&1 &
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.4 > srun-v${NUM}-4.log 2>&1 &

# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.5 > srun-v${NUM}-5.log 2>&1 &

# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 -w node06,node07 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.5 > srun-v${NUM}-5.log 2>&1 &

