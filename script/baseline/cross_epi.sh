exp_fold="1 2 3 4 5"
data_fold="1 2 3 4 5"
exp_cell_line=$1
test_cell_line=$2
gpu=$3
export CUDA_VISIBLE_DEVICES=$gpu
model="sage2"
t=0.4
nodes=700
layer=1
hidden=1024
dropout=0.6
lr=1e-2
task="regression"

for exp in ${exp_fold};do
       python experiment.py --experiment graph_${task}_${exp_cell_line}_${exp}_lr$4 \
--model ${model} \
--cell_line ${test_cell_line} \
--fold ${exp} \
--task ${task} \
--hidden ${hidden} \
-t ${t} \
--dropout ${dropout} \
--layers ${layer} \
--lr ${lr} \
--nodes ${nodes} \
--patience 10 \
--build mm10 \
--mode test
done
