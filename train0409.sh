NUM=10
# sleep 5s && srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 kmeans-32 > srun-v${NUM}-0.log 2>&1 &
sleep 5s && srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 kmeans-64 > srun-v${NUM}-0.log 2>&1 &
sleep 5s && srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 kmeans-100 > srun-v${NUM}-0.log 2>&1 &
sleep 5s && srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 kmeans-train-32 > srun-v${NUM}-0.log 2>&1 &
sleep 5s && srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 kmeans-train-64 > srun-v${NUM}-0.log 2>&1 &
sleep 5s && srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 kmeans-train-100 > srun-v${NUM}-0.log 2>&1 &
sleep 5s && srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 kmeans-valid-32 > srun-v${NUM}-0.log 2>&1 &
sleep 5s && srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 kmeans-valid-64 > srun-v${NUM}-0.log 2>&1 &
sleep 5s && srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 kmeans-valid-100 > srun-v${NUM}-0.log 2>&1 &
