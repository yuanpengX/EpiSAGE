datadir="/data2/xiongyuanpeng/gene_expression/data/heart/"
for i in `seq 1 19`;do
    python bined_data.py 100 chr${i} H3K27ac ${datadir} ;
    python extract_feature.py chr${i} ${datadir} ;
done