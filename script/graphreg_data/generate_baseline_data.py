import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle as pkl
# 提取的是基本的数据集
# gene annotation
data_dir='/data2/xiongyuanpeng/gene_expression/data/'
anno_file='gencode.vM9.annotation.gff3' # use mm10 annotation for coordinates

data = {'start':[],'end':[],'chr':[], 'strand':[],'id':[]}
for line in open(data_dir+anno_file):
    if not line.startswith('#'):
        line = line.strip().split('\t')
        if line[2] == 'gene':
            anno=line[-1]
            ids = re.findall(r'gene_id=(.*?);',anno)[0].split('.')[0]
            data['start'].append(line[3])
            data['end'].append(line[4])
            data['strand'].append(line[6])
            data['chr'].append(line[0])
            data['id'].append(ids)
gff = pd.DataFrame(data)
gene_count = '/mesc/GSM2375118_ESC_RNASeq_WT_Rep1.tagcount.txt'
print(len(gff))
data = {'id':[],'count':[]}
with open(data_dir+gene_count) as fp:
    for line in fp:
        ids, count = line.strip().split()
        data['id'].append(ids.split('.')[0])
        data['count'].append(count)
rna_seq=pd.DataFrame(data)
result = pd.merge(gff,rna_seq,on='id',how='inner')
print(len(result))
result.to_csv('/data2/xiongyuanpeng/gene_expression/data/mesc/processed/baseline_data/annotation.csv',index=False)

# 分染色体选择特征数据
resolution = 100

def get_tss_tts(row):
    start = int(row['start'])//resolution
    end = int(row['end'])//resolution
    strand = row['strand']

    if strand=='+':
        return int(start),int(end)
    else:
        return int(end),int(start)

for i in range(1,20):
    print(f'chr{i}')
    chr_data = result[result['chr']==f'chr{i}']
    fp1 = [float(v.strip().split()[1]) for v in open(data_dir+f'/mesc/processed/H3K27Ac_chr{i}.bedgraph')]
    fp2 = [float(v.strip().split()[1]) for v in open(data_dir+f'/mesc/processed/H3K27me3_chr{i}.bedgraph')]
    fp3 = [float(v.strip().split()[1]) for v in open(data_dir+f'/mesc/processed/H3K4me1_chr{i}.bedgraph')]
    fp4 = [float(v.strip().split()[1]) for v in open(data_dir+f'/mesc/processed/H3K4me3_chr{i}.bedgraph')]
    fp5 = [float(v.strip().split()[1]) for v in open(data_dir+f'/mesc/processed/H3K9me3_chr{i}.bedgraph')]
    for idx, row in tqdm(chr_data.iterrows()):
        try:
            tss, tts = get_tss_tts(row)

            # 提取特征维度
            feat1 = fp1[tss - 50:tss+50]
            feat2 = fp2[tss - 50:tss+50]
            feat3 = fp3[tss - 50:tss+50]
            feat4 = fp4[tss - 50:tss+50]
            feat5 = fp5[tss - 50:tss+50]
            tss_feat = np.vstack([feat1, feat2, feat3, feat4, feat5])
            assert tss_feat.shape == (5,100)

            # 提取tts特征
            s,e = tts - 20, tts+20
            feat1 = fp1[s:e]
            feat2 = fp2[s:e]
            feat3 = fp3[s:e]
            feat4 = fp4[s:e]
            feat5 = fp5[s:e]

            tts_feat = np.vstack([feat1, feat2, feat3, feat4, feat5])
            assert tts_feat.shape == (5,40)
            chrs = row['chr']
            data = (tss_feat, tts_feat, int(row['count']))
            ids = row['id']
            pkl.dump(data, open(data_dir+f'mesc/processed/baseline_data/{chrs}_{ids}.bin','wb'))
        except:
            print(tts, tss)
            continue