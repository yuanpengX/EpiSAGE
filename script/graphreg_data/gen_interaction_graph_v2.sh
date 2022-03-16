cell_line=$1
ref_dir="/data2/xiongyuanpeng/gene_expression/data/"
processed_dir="/data2/xiongyuanpeng/gene_expression/data/${cell_line}/processed/"
result_dir="/data2/xiongyuanpeng/gene_expression/data/${cell_line}/processed/baseline_data/"

for i in `seq 1 19`;do
    python /data2/xiongyuanpeng/gene_expression/script/graphreg_data/3_intrachrome_interaction_v2.py ${ref_dir} ${processed_dir} ${result_dir} chr${i} ;
done
