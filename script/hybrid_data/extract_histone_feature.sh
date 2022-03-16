ts="H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me3"
#datadir="/data2/xiongyuanpeng/gene_expression/data/kidney/"
outfile="${datadir}/processed/splited_histone_modification/"

for i in `seq 1 19`;
do
    for t in ${ts};
    do
        (
            echo ${t} "chr${i}" ;
            python extract_histone_feature.py chr${i} ${t} ${datadir} ;
        )&
    done
    wait
done
echo "extract histone feature done!";