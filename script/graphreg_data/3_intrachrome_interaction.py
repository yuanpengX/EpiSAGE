import cooler
import numpy as np
from tqdm import tqdm
hic_dir='/data/group/data/gene_expression/data/heart10.5/'
hic_name = 'GSM2544836_KR.cool'
c = cooler.Cooler(hic_dir+hic_name)
gene_dir='/data2/xiongyuanpeng/gene_expression/data/heart/processed/splited_gene_expression/'
output_dir = '/data2/xiongyuanpeng/gene_expression/data/heart/processed/feature/interaction_matrix/'

resolution = 5000
length = 1200 // 2
for i in range(2,20):
    chrs = f'chr{i}'

    for gene in tqdm(open(gene_dir+f'{chrs}_anno.sorted.gtf')):
        gene = gene.strip().split()
        start = int(gene[3])
        end = int(gene[4])
        gene_id = gene[-1]
        index = ((start + end)//2)//resolution
        start_index = (index - length) * resolution
        end_index = (index + length) * resolution
        try:
            mat = c.matrix(balance=True).fetch(f'chr1:{start_index}-{end_index}')
            assert mat.shape == (1200, 1200)
        except:
            continue
        # save the interaction matrix
        np.save(f'{output_dir}/{chrs}_{gene_id}.npy', mat)