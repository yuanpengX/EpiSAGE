# input
cell_line=$2
outdir="/data2/xiongyuanpeng/gene_expression/data/${cell_line}/processed/"
scriptdir="/data2/xiongyuanpeng/gene_expression/script/graphreg_data/"
file=${outdir}/$1.bed
outfile=${outdir}/$1.bined.bedgraph
python ${scriptdir}/1_bined_data_v2.py ${file} ${outfile} 100 ;
types=$1
for i in `seq 1 19`;
do
    (
    awk -v chr=chr${i} '{if($1==chr){print $0}}' ${outfile} > ${outdir}/${types}_chr${i}.bedgraph ;
    )&
done
