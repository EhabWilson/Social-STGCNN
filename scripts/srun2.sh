# python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth --tag  eth --use_lrschd --num_epochs 250 --gpu_num 3 && echo "eth Launched." &
# P0=$!

# python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset hotel --tag hotel --use_lrschd --num_epochs 250 --gpu_num 0 && echo "hotel Launched." &
# P1=$!

# python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset univ --tag univ --use_lrschd --num_epochs 250 --gpu_num 0 && echo "univ Launched." &
# P2=$!

# python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara1 --tag zara1 --use_lrschd --num_epochs 250 --gpu_num 0 && echo "zara1 Launched." &
# P3=$!

# python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara2 --tag zara2 --use_lrschd --num_epochs 250 --gpu_num 0 && echo "zara2 Launched." &
# P4=$!

python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth --tag  eth --use_lrschd --num_epochs 250 && echo "eth Launched." & \
python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset hotel --tag hotel --use_lrschd --num_epochs 250 && echo "hotel Launched." & \
python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset univ --tag univ --use_lrschd --num_epochs 250 && echo "univ Launched." & \
python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara1 --tag zara1 --use_lrschd --num_epochs 250 && echo "zara1 Launched." & \
python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara2 --tag zara2 --use_lrschd --num_epochs 250 && echo "zara2 Launched." &