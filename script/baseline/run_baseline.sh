fold=$3
gpu=$4
export CUDA_VISIBLE_DEVICES=${gpu}
model=$1
cell_line=$2
task="regression"
python experiment.py --experiment ${task}_baseline_${cell_line}_${fold} \
--model ${model} \
--cell_line ${cell_line} \
--fold ${fold} \
--task ${task} \
--batch_size 256 \
--patience 10