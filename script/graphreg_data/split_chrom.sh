#file="GSM2375127_mm10_ESC_H3K27Ac_WT_Rep1.bined.bedgraph"
#outfile="H3K27Ac"
#file="GSM2375147_mm10_ESC_H3K4me3_WT_Rep1.bined.bedgraph"
#outfile="H3K4me3"
#file="GSM3188172_mm10.CAGE_UFR3_SCR_ESC_exp_1.bined.bedgraph"
#outfile="CAGEseq"
file="GSM5117383_ATACseq_WT_fresh_rep1.bined.bedgraph"
outfile="ATACseq"
outdir='/data2/xiongyuanpeng/gene_expression/data/mesc/processed/'
for i in `seq 1 19`;
do
    (
    awk -v chr=chr${i} '{if($1==chr){print $0}}' ${outdir}/${file} > ${outdir}/${outfile}_chr${i}.bedgraph ;
    )&
done