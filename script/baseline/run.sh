#python experiment.py -m attenchrome -e baseline_mesc &
cell_line="mesc"
threshold=8
node=400
dropouts="0.9 0.8 0.7 0.6"
lr="1e-2"
layers="2 3 4 5"
hidden=1024
for dropout in $dropouts;do
    count=-1
    for layer in ${layers};do
    count=$[count+1]
    echo ${layer} ${dropout}
    export CUDA_VISIBLE_DEVICES=${count}
    (
    python experiment.py --hidden ${hidden} -m sage -e graph_${cell_line}_${lr}_d${dropout}_l${layer} --layers ${layer} -d ${dropout} --lr ${lr} -n ${node} -t ${threshold} -c ${cell_line};
    )&
    done
    wait
done

