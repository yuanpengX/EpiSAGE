ts="dnase"
#"H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me3"
#datadir="/data2/xiongyuanpeng/gene_expression/data/heart/"
outfile="${datadir}/processed/splited_histone_modification/"
for t in ${ts};
do
    echo ${t} ;
    file="${datadir}/processed/${t}.bed" ;
    echo "split data" ${file} ;
    for i in `seq 1 19`;
    do
        nohup awk -v chr=chr${i} '{if($1==chr){print $0}}' ${file} > ${outfile}/${t}_chr${i}.bed &
    done
done
echo "split histone data done"