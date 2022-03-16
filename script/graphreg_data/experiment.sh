export CUDA_VISIBLE_DEVICES=1
experiment="graph"
model="graph"
hidden=128
dropout=0.5
fold=1
activation=relu
learning_rate="1e-3"
patience=50
head=4
# only valid for graphsage
layers=3
threshold=3
aggregator_type=mean

export CUDA_VISIBLE_DEVICES=1
model='cnn'
experiment='cnn_mseloss'
python experiment.py --head ${head} -e ${experiment}_${hidden} -m ${model} --hidden ${hidden} -d ${dropout} -l ${layers} -t ${threshold} -f ${fold} --ag ${aggregator_type} --act ${activation} --lr ${learning_rate} -p ${patience} &

export CUDA_VISIBLE_DEVICES=1
model='graph'
experiment='graph_mseloss'
python experiment.py --head ${head} -e ${experiment}_${hidden} -m ${model} --hidden ${hidden} -d ${dropout} -l ${layers} -t ${threshold} -f ${fold} --ag ${aggregator_type} --act ${activation} --lr ${learning_rate} -p ${patience} &

export CUDA_VISIBLE_DEVICES=2
hiddens="64 128 256 512"
model='sage'
experiment='graphsage_mseloss'
for hidden in ${hiddens};do
(
    python experiment.py --head ${head} -e ${experiment}_${hidden} -m ${model} --hidden ${hidden} -d ${dropout} -l ${layers} -t ${threshold} -f ${fold} --ag ${aggregator_type} --act ${activation} --lr ${learning_rate} -p ${patience})&
done