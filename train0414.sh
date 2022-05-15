NUM=12
NAME="$NUM-3"
srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 > srun-v${NAME}.log 2>&1 &
# srun -J v${NUM} -N 1 -p RTX3090 -w node06 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} 0.0 kmeans-32 5 > srun-v${NUM}-5.log 2>&1 &
NAME="$NUM-7"
srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 7 > srun-v${NAME}.log 2>&1 &



# for REP in 3 5 7; do
#     NAME="$NUM-$REP"
#     srun -J v${NUM} -N 1 -p RTX3090 -w node06 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 7 > srun-v${NAME}.log 2>&1 &
# done

# srun: error: Required nodelist includes more nodes than permitted by max-node count (2 > 1). Eliminating nodes from the nodelist.

# -x master node04,node09,master

# srun -J v10-test -N 1 -p RTX2080Ti --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/kmeans-100 --model social_stgcnn10 > srun-100.log 2>&1 & 