import cooler
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl
import sys

chrs = sys.argv[1]
cell_line = 'mesc'
hic_dir= f'/data2/xiongyuanpeng/gene_expression/data/{cell_line}/'
hic_name = 'GSE171749_HiC_WT_5000_KR.cool'
c = cooler.Cooler(hic_dir+hic_name)
#
datadir='/data2/xiongyuanpeng/gene_expression/data/mesc/processed/'
fp1 = open(datadir+f'H3K27Ac_{chrs}.bedgraph').readlines()
fp2 = open(datadir+f'H3K4me3_{chrs}.bedgraph').readlines()
fp3 = open(datadir+f'ATACseq_{chrs}.bedgraph').readlines()

fp4 = open(datadir+f'CAGEseq_{chrs}.bedgraph').readlines()
# 普通特征
s1 = int(0)
step1 = int(6e6//100)

# cage和相互作用图特征
s2 = int(0)
step2 = int(6e6//5000)

count = 0
while True:
    e1 = s1 + step1
    e2 = s2 + step2
    try:
        feat1 = np.array([float(v.strip().split()[-1]) for v in fp1[s1:e1]])
        feat2 =np.array([float(v.strip().split()[-1]) for v in fp2[s1:e1]])
        feat3 =np.array([float(v.strip().split()[-1]) for v in fp3[s1:e1]])
        label = np.array([float(v.strip().split()[-1]) for v in fp4[s2:e2]])

        start_index = s1 * 100
        end_index = e1 * 100
        mat = c.matrix(balance=True).fetch(f'chr1:{start_index}-{end_index}')

        #print(feat1.shape)
        assert len(feat1) == step1
        #print(feat2.shape)
        assert len(feat2) == step1
        #print(feat3.shape)
        assert len(feat3) == step1
        #print(label.shape)
        assert len(label) == step2
        assert mat.shape == (step2, step2)
        #print('hi2')
        pkl.dump((feat1, feat2, feat3, label, mat), open(f'{datadir}/{chrs}_{count}.bin','wb'))
        count+=1
        #print('hi1')
        s1 = s1 + int(2e6//100)
        s2 = s2 + int(2e6//5000)

    except Exception as e:
        print(s1,e1,s2,e2)
        print(feat1.shape, feat2.shape, feat3.shape, label.shape, mat.shape)
        break