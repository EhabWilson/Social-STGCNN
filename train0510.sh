# to test how much gain from baseline

# NUM=9
# NAME="$NUM-2xlr"
# # srun -J v${NUM} -N 1 -p RTX2080TI --gres gpu:1 --priority 9999999 
# bash ./scripts/train-v3.sh social_stgcnn${NUM} ${NAME} 0.02 >srun-v${NAME}.log 2>&1 &


# NUM=9
# NAME="$NUM-1.5xlr"
# # srun -J v${NUM} -N 1 -p RTX2080TI --gres gpu:1 --priority 9999999 
# bash ./scripts/train-v3.sh social_stgcnn${NUM} ${NAME} 0.015 >srun-v${NAME}.log 2>&1 &

# NUM=9
# NAME="$NUM-3xlr"
# # srun -J v${NUM} -N 1 -p RTX2080TI --gres gpu:1 --priority 9999999 
# bash ./scripts/train-v3.sh social_stgcnn${NUM} ${NAME} 0.03 >srun-v${NAME}.log 2>&1 &

# test
# NUM=9
# NAME="$NUM-2xlr"
# # srun -J v${NUM} -N 1 -p RTX2080TI --gres gpu:1 --priority 9999999 
# # bash ./scripts/train-v3.sh social_stgcnn${NUM} ${NAME} 0.02 >srun-v${NAME}.log 2>&1 &
# python testmodel.py --log_dir runs/${NAME}

# NUM=9
# NAME="$NUM-1.5xlr"
# # srun -J v${NUM} -N 1 -p RTX2080TI --gres gpu:1 --priority 9999999 
# # bash ./scripts/train-v3.sh social_stgcnn${NUM} ${NAME} 0.015 >srun-v${NAME}.log 2>&1 &
# # python testmodel.py --log_dir runs/${NAME}

# NUM=9
# NAME="$NUM-3xlr"
# # srun -J v${NUM} -N 1 -p RTX2080TI --gres gpu:1 --priority 9999999 
# bash ./scripts/train-v3.sh social_stgcnn${NUM} ${NAME} 0.03 >srun-v${NAME}.log 2>&1 &
# # python testmodel.py --log_dir runs/${NAME}

# NUM=9
# NAME="$NUM-0.5xlr"
# # srun -J v${NUM} -N 1 -p RTX2080TI --gres gpu:1 --priority 9999999 
# bash ./scripts/train-v3.sh social_stgcnn${NUM} ${NAME} 0.005 >srun-v${NAME}.log 2>&1 &
# # python testmodel.py --log_dir runs/${NAME}

# NUM=9
# NAME="$NUM-0.8xlr"
# # srun -J v${NUM} -N 1 -p RTX2080TI --gres gpu:1 --priority 9999999 
# bash ./scripts/train-v3.sh social_stgcnn${NUM} ${NAME} 0.008 >srun-v${NAME}.log 2>&1 &
# # python testmodel.py --log_dir runs/${NAME}

NUM=9
NAME="$NUM-2.5xlr"
# srun -J v${NUM} -N 1 -p RTX2080TI --gres gpu:1 --priority 9999999 
bash ./scripts/train-v3.sh social_stgcnn${NUM} ${NAME} 0.025 >srun-v${NAME}.log 2>&1 &
# python testmodel.py --log_dir runs/${NAME}




# NUM=29
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX2080Ti --gres gpu:1 --priority 9999999 bash ./scripts/train-v1-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3  >srun-v${NAME}.log 2>&1 &
