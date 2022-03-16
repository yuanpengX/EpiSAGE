cell_line="liver"
basedir="/data2/xiongyuanpeng/gene_expression/data/${cell_line}/"
datadir="${basedir}/processed/"
result_dir="datadir/baseline_data/"

hicConvertFormat --matrices ${basedir}/GSM5410977_WT-1_hic.allValidPairs.hic -o ${datadir}/GSM541097.cool --inputFormat hic --outputFormat cool --resolutions 5000
# cool to cool to get normalize data
hicConvertFormat -m ${datadir}/GSM541097_5000.cool --inputFormat cool --outputFormat cool -o ${datadir}/GSM541097_KR.cool --correction_name KR
# convert cool to ginteractions format
hicConvertFormat --m ${datadir}/GSM541097_KR.cool --outFileName KR.ginteractions --inputFormat cool --outputFormat ginteractions
# extract chr1 information
