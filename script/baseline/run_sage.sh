cell_line=$1
fold=$2
gpu=$3
export CUDA_VISIBLE_DEVICES=${gpu}
model="sage2"
t=0.4
nodes=700
layer=1
hidden=1024
dropout=0.6
lr=1e-2
task="classification"

python experiment.py --experiment graph2_${task}_${cell_line}_${fold}_lr$4 \
--model ${model} \
--cell_line ${cell_line} \
--fold ${fold} \
--task ${task} \
--hidden ${hidden} \
-t ${t} \
--dropout ${dropout} \
--layers ${layer} \
--lr ${lr} \
--nodes ${nodes} \
--patience 10 \
--build mm10 \
--mode train