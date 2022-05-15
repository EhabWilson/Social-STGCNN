# command

NUM=24
for act in nn.PReLU nn.GELU nn.Sigmoid nn.ReLU nn.SiLU; do
    NAME="$NUM-$act"
    srun -J v${NUM} -N 1 -p RTX2080Ti --gres gpu:1 --priority 9999999 python testmodel_v1.py --log_dir ./runs/${NAME} > srun-TEST${NAME}.log 2>&1 &
done