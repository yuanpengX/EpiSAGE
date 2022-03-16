datadir=/data/group/data/gene_expression/data/heart10.5/
# first extract cool file from hic data
hicConvertFormat --matrices GSM2544836_No_Tx.hic -o 5000.cool --inputFormat hic --outputFormat cool --resolutions 5000
# cool to cool to get normalize data
hicConvertFormat -m 5000.cool --inputFormat cool --outputFormat cool -o KR.cool --correction_name KR
# convert cool to ginteractions format
hicConvertFormat --m processed/KR.cool --outFileName processed/baseline_data/KR.ginteractions --inputFormat cool --outputFormat ginteractions
# extract chr1 information

for i in `seq 1 19`;
do
    awk -v chr=chr${i} '{if($1==chr && $4==chr){print $0}}' ${datadir}/GSM2544836_KR.ginteractions.tsv > ${datadir}/processed/GSM2544836_KR_chr${i}.ginteractions
done