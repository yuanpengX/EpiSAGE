import os
import pandas as pd
import sys
from glob import glob
refdir='/data/group/data/gene_expression/data/ref/'
reffile = 'gencode.vM21.annotation.gff3'
gene2pos = {}

for line in open(refdir+reffile):
    if line.startswith('#'):
        continue
    else:
        lines = line.split()
        if lines[2]=='gene':
            gene = lines[-1].split(';')[0].split('=')[-1]
            gene2pos[gene] = '\t'.join(lines[:5] +[gene,])

# convert to default setting
datadir=sys.argv[1]
datafile = '*.tsv'
data = pd.read_csv(glob(datadir+datafile)[0],sep='\t')
# 写两个文件
# 一个是基因标签数据
# 一个是所有的基因数据

with open(datadir+'processed/gene_annotation.tsv','w') as fp:
    for gene_id,pos in gene2pos.items():
        fp.write(gene2pos[gene_id]+f'\n')
with open(datadir+'processed/rna_seq.tsv','w') as fp:
    for idx, row in data.iterrows():
        gene_id = row['gene_id']
        if gene_id in gene2pos:
            fp.write(gene2pos[gene_id]+f'\t{row["FPKM"]}\n')

