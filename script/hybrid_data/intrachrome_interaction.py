import sys
import numpy as np
import pickle as pkl

ref_dir=sys.argv[1]#'/data/group/data/gene_expression/data/ref/'
processed_dir=sys.argv[2]#'/data/group/data/gene_expression/data/heart10.5/processed/'
result_dir = sys.argv[3]
hic_dir = sys.argv[4]
chrs = sys.argv[5]#'chr1'
# 先根据染色体大小，将基因映射到index中，获得基因到染色质的map
chrom2size = {line.strip().split()[0]:line.strip().split()[1] for line in open(ref_dir+'mm10.chrom.sizes')}

resolution = 5000
size = int(chrom2size[chrs])
max_len = size // resolution + 1
# assign every gene to bins
gene_id2chrom_index = {}
chrom_index2gene_id = {}
with open(processed_dir+f'splited_gene_expression/{chrs}_anno.sorted.gtf') as fp:
    for line in fp:
        lines = line.strip().split('\t')
        start = int(lines[3])
        end = int(lines[4])

        anno = lines[5]#lines[8].split()[1][1:-2] + '_' + lines[5]

        index = (start+ end)//(2 * resolution)

        gene_id2chrom_index[anno] = index
        chrom_index2gene_id[index] = anno
print(len(gene_id2chrom_index))
# 构造基因array
gene_id2array_index = {gene_id:idx for (idx, gene_id) in enumerate(gene_id2chrom_index)}
array_index2gene_id = {value:key for key,value in gene_id2array_index.items()}

gene_num = len(gene_id2array_index)

# norm 数据
#hic_dir=f'/data/group/data/gene_expression/data/IMR90/5kb_resolution_intrachromosomal/{chrs}/MAPQGE30/'
#norm_vec = [float(value) for value in open(hic_dir+f'{chrs}_5kb.KRnorm')]

# 处理hic raw数据
count = 0
interaction_array = np.zeros((gene_num, gene_num))
for line in open(hic_dir + f'GSM2544836_KR_{chrs}.ginteractions'):
    lines = line.strip().split()
    first, second, value = lines[2], lines[5],lines[-1]
    first = int(first)
    second = int(second)
    value= float(value)

    # 确认两个都有基因
    first_index = first // resolution - 1
    second_index = second//resolution - 1
    if first_index in chrom_index2gene_id and second_index in chrom_index2gene_id:
        if first_index == second_index:
            count +=1
        # norm value
        #normed_value = value/(norm_vec[first_index+1]* norm_vec[second_index+1])
        normed_value = value
        index1 = gene_id2array_index[chrom_index2gene_id[first_index]]
        index2 = gene_id2array_index[chrom_index2gene_id[second_index]]
        # 对称问题
        interaction_array[index1, index2]+= normed_value
        interaction_array[index2, index1]+= normed_value

# store datap
#result_dir='/data/group/data/gene_expression/data/heart10.5/processed/hic_array/'

pkl.dump(gene_id2chrom_index, open(result_dir+f'{chrs}_gene_id2chrom_index.bin','wb'))
pkl.dump(chrom_index2gene_id, open(result_dir+f'{chrs}_chrom_index2gene_id.bin','wb'))
pkl.dump(gene_id2array_index, open(result_dir+f'{chrs}_gene_id2array_index.bin','wb'))
pkl.dump(array_index2gene_id, open(result_dir+f'{chrs}_array_index2gene_id.bin','wb'))
pkl.dump(interaction_array, open(result_dir+f'{chrs}_interaction_array.bin','wb'))