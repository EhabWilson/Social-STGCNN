# to test how much gain from baseline

# NUM=9
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 >srun-v${NAME}.log 2>&1 &

# NUM=15
# for drop in 0.1 0.3; do 
# # for drop in 0.0 0.1 0.2 0.3 0.4; do 
#     NAME="$NUM-3-$drop"
#     srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 -w node07 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} ${drop} kmeans-valid-32 3 >srun-v${NAME}-${drop}.log 2>&1 &
# done

# "nn.PReLU"
# "nn.GELU"
# "nn.Sigmoid"
# "nn.ReLU"
# "nn.SiLU"

# NUM=21
# for kernel in 3 5 7; do 
#     NAME="$NUM-$kernel"
#     srun -J v${NUM} -N 1 -p RTX2080Ti --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-valid-32 ${kernel} >srun-v${NAME}.log 2>&1 &
# done

# NUM=22
# for drop in 0.0 0.1 0.2; do 
#     NAME="$NUM-$drop"
#     srun -J v${NAME} -N 1 -p RTX2080Ti --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} ${drop} kmeans-valid-32 3 > srun-v${NAME}-${drop}.log 2>&1 &
# done

# NUM=23
# for drop in 0.0 0.3; do 
# # for drop in 0.0 0.1 0.2 0.3 0.4; do 
#     NAME="$NUM-$drop"
#     srun -J v${NAME} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} ${drop} kmeans-valid-32 3 > srun-v${NAME}-${drop}.log 2>&1 &
# done

# NUM=24
# for act in nn.PReLU nn.GELU nn.Sigmoid nn.ReLU nn.SiLU; do
#     NAME="$NUM-$act"
#     srun -J v${NAME} -N 1 -p RTX2080Ti --gres gpu:1 --priority 9999999 bash ./scripts/train-v1-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-valid-32 3 ${act} > srun-v${NAME}.log 2>&1 &
# done

# NUM=25
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 > srun-v${NAME}.log 2>&1 &

# NUM=26
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v0-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 >srun-v${NAME}.log 2>&1 &

# NUM=27
# for act in nn.PReLU nn.GELU nn.Sigmoid nn.ReLU nn.SiLU; do
#     NAME="$NUM-$act"
#     srun -J v${NAME} -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 bash ./scripts/train-v1-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-valid-32 3 ${act} >srun-v${NAME}.log 2>&1 &
# done

# NUM=28
# for drop in 8 10 12 14 16; do 
# # for drop in 0.0 0.1 0.2 0.3 0.4; do 
#     NAME="$NUM-$drop"
#     srun -J v${NAME} -N 1 -p RTX2080Ti --gres gpu:1 --priority 9999999 bash ./scripts/train-v2-repeat.sh social_stgcnn${NUM} ${NAME} ${drop} kmeans-valid-32 3 nn.Sigmoid > srun-v${NAME}.log 2>&1 &
# done


# NUM=29
# NAME="$NUM-3"
# srun -J v${NUM} -N 1 -p RTX2080Ti --gres gpu:1 --priority 9999999 bash ./scripts/train-v1-repeat.sh social_stgcnn${NUM} ${NAME} 0.0 kmeans-32 3 nn.Sigmoid >srun-v${NAME}.log 2>&1 &
