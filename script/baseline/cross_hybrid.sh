exp_fold="1 2 3 4 5"
data_fold="1 2 3 4 5"
exp_cell_line=$1
test_cell_line=$2
gpu=$3
export CUDA_VISIBLE_DEVICES=$gpu
model="hybrid"
task="regression"
for exp in ${exp_fold};do
        python experiment.py --batch_size 256 --experiment ${task}_baseline_${exp_cell_line}_${exp} --model ${model} --mode test --task regression --cell_line ${test_cell_line} --fold ${exp} --build mm10
done
