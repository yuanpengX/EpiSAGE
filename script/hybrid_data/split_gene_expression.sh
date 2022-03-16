#datadir="/data2/xiongyuanpeng/gene_expression/data/kidney/"
#echo "zip data"
#gzip -d "${datadir}*.gz";
mkdir "${datadir}/processed/"
file="${datadir}/processed/rna_seq.tsv"
gene_annotation="${datadir}/processed/gene_annotation.tsv"
outdir="${datadir}/processed/splited_gene_expression/"
mkdir ${outdir}
python process_rna_seq.py ${datadir} ;
for i in `seq 1 19`;
do
    awk -v chr=chr${i} '{if($1==chr){print $0}}' ${file} > ${outdir}/chr${i}.gtf ;
    awk -v chr=chr${i} '{if($1==chr){print $0}}' ${gene_annotation} > ${outdir}/chr${i}_anno.gtf ;
    sort ${outdir}/chr${i}.gtf  -n -k 4 -t $'\t' > ${outdir}/chr${i}.sorted.gtf ;
    sort ${outdir}/chr${i}_anno.gtf  -n -k 4 -t $'\t' > ${outdir}/chr${i}_anno.sorted.gtf ;
done
