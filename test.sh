# command

# srun -J v10-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/12-3  > srun-12-3.log 2>&1 &
# srun -J v10-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/12-5  > srun-12-5.log 2>&1 &
# srun -J v10-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/12-7  > srun-12-7.log 2>&1 &

# srun -J v9-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir runs/9-3.8177.18488.32315  > srun-9-3.log 2>&1 &

# NAME=13-1
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &
# NAME=13-2
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &


# NAME=14-0.0-1
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &
# NAME=14-0.0-2
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &
# NAME=14-0.3-1
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &
# NAME=14-0.3-2
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &

# NAME=15-0.0-1
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &
# NAME=15-0.0-2
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &
# NAME=15-0.2-1
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &
# NAME=15-0.2-2
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &

# NAME=16-1
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &
# NAME=16-2
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 python testmodel.py --log_dir ./runs/${NAME}  > srun-${NAME}.log 2>&1 &

NAME=9-3.8177
# srun -J v${NAME}-test -N 1 -p RTX3090 --gres gpu:1 --priority 9999999 
# python testmodel_v2.py --log_dir ./runs/${NAME}  > srun-test-${NAME}.log 2>&1
python testmodel.py --log_dir ./runs/${NAME}  > srun-test-${NAME}.log 2>&1
