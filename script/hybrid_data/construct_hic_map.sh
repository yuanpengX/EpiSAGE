ref_dir="/data/group/data/gene_expression/data/ref/"
processed_dir="${data_dir}/processed/"
result_dir="/data2/xiongyuanpeng/gene_expression/data/hic_array/"
mkdir ${result_dir}
hic_dir="/data/group/data/gene_expression/data/heart10.5/processed/"
#chrs="1"
for chrs in `seq 1 19`;do
    python intrachrome_interaction.py ${ref_dir} ${processed_dir} ${result_dir} ${hic_dir} chr${chrs}
done