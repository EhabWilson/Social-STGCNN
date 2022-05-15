# to test how much gain from baseline
# NUM=
# NAME="basline-3"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 >srun-v${NAME}.log 2>&1 &

# NUM=9
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 >srun-v${NAME}.log 2>&1 &

# NUM=10
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 >srun-v${NAME}.log 2>&1 &

# NUM=12
# for kernel in 5 7; do
#     NAME="$NUM-$kernel"
#     srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-valid-32 ${kernel} > srun-v${NAME}.log 2>&1 &
# done
# use y as cluster initialization 11

# NUM=11
# NAME="$NUM-3-2"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 32 3 >srun-v${NAME}.log 2>&1 &

# NUM=14
# for drop in 0.0 0.1 0.2 0.3 0.4 0.5; do 
#     NAME="$NUM-3-$drop"
#     srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} ${drop} kmeans-valid-32 3 >srun-v${NAME}-${drop}.log 2>&1 &
# done

NUM=15
for drop in 0.1 0.3; do 
# for drop in 0.0 0.1 0.2 0.3 0.4; do 
    NAME="$NUM-3-$drop"
    srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 -w node07 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} ${drop} kmeans-valid-32 3 >srun-v${NAME}-${drop}.log 2>&1 &
done

# NUM=17
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 >srun-v${NAME}.log 2>&1 &

# NUM=18
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 >srun-v${NAME}.log 2>&1 &


# NUM=19
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 >srun-v${NAME}.log 2>&1


# NUM=20
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 >srun-v${NAME}.log 2>&1 &


# for REP in 3 5 7; do
#     NAME="$NUM-$REP"
#     srun -J v${NUM} -N 1 -p RTX3090 -w node06 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 7 > srun-v${NAME}.log 2>&1 &
# done

# srun: error: Required nodelist includes more nodes than permitted by max-node count (2 > 1). Eliminating nodes from the nodelist.

# -x master node04,node09,master

# srun -J v10-test -N 1 -p RTX2080Ti --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/kmeans-100 --model social_stgcnn10 > srun-100.log 2>&1 &
