cell_lines="mesc heart liver nural"
for fold in `seq 1 10`;do
    count=0
    for cell_line in ${cell_lines};do
        export CUDA_VISIBLE_DEVICES=$count
        echo ${cell_line} ${fold}
        count=$[count+1]
        (
        python experiment.py -m deepchrom -e baseline_mesc --cell_line ${cell_line} --fold ${fold}
        )&
    done
    wait
done