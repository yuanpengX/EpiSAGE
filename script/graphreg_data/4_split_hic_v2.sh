datadir="/data2/xiongyuanpeng/gene_expression/data/$1/processed/"
for i in `seq 1 19`;
do
    awk -v chr=chr${i} '{if($1==chr && $4==chr){print $0}}' ${datadir}/KR.ginteractions.tsv > ${datadir}/baseline_data/KR_chr${i}.ginteractions
done