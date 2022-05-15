begin=0
end=0
para=1
parallel_per_gpu=5
# export debug=1
# export exp_name=ethucy
. looptrain.sh

envs=(
    "eth"
    "hotel"
    "univ"
    "zara1"
    "zara2"
)

# for seed in {0,1,2,3,4,}; do
for env in ${envs[*]}; do
run --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset ${env} --tag ${env} --use_lrschd --num_epochs 250
sleep 10
done
# done
wait

# if [ $? -eq 0 ]; then
# 	curl "https://api.day.app/RwG2pGqj8DLE8MtJaGCbod/Finished/($exp_name)"
# else
# 	curl "https://api.day.app/RwG2pGqj8DLE8MtJaGCbod/Failed/($exp_name)"
# fi
