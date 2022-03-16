from tqdm import tqdm
import os

processed_dir = '/data2/xiongyuanpeng/gene_expression/data/heart/processed/'
gene_dir=f'{processed_dir}/splited_gene_expression/'
hic_dir = f'{processed_dir}/feature/interaction_matrix/'

for i in range(1,20):
    chrs = f'chr{i}'
    fpo = open('/data2/xiongyuanpeng/gene_expression/data/heart/processed/' + f'dataset_{chrs}.txt','w')
    for gene in tqdm(open(gene_dir+f'{chrs}.sorted.gtf')):
        gene = gene.strip().split()
        gene_id = gene[-2]

        label = float(gene[-1])

        feature_name = f'{processed_dir}/feature/histone/{chrs}_{gene_id}.npy'
        matrix_name = f'{processed_dir}/feature/interaction_matrix/{chrs}_{gene_id}.npy'
        if os.path.isfile(feature_name) and os.path.isfile(matrix_name):
            fpo.write(f'{feature_name}\t{matrix_name}\t{label}\n')


