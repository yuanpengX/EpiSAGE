exp_fold="1 2 3 4 5"
data_fold="1 2 3 4 5"
exp_cell_line=$1
test_cell_line=$2
gpu=$3
export CUDA_VISIBLE_DEVICES=$gpu
model="sage2"
for exp in ${exp_fold};do
    for data in ${data_fold};do
        python experiment.py -e graph_${exp_cell_line}_${exp} -m ${model} --mode test --task regression --cell_line liver --fold ${data}
done
done
