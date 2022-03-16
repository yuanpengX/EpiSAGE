import sys
from sklearn import ensemble
import numpy as np

# 思路

# 对于每一个基因，抽取他上下游一共6M的数据

chrs = sys.argv[1]
data_dir = sys.argv[2]

resolution = 100 # 100bp resolution
ranges = 6e6 # 6Mb
# 跨cell line有个比较大的问题是，
# 数据需要normalize，好在可以在batch内部做

# 预先读取特征文件
features = ['H3K4me3', 'H3K27ac', 'dnase']
feature_file = {key:[float(v) for v in open(data_dir+ f'/processed/splited_histone_modification/{key}_{chrs}_100.bed').readlines()] for key in features}

with open(data_dir+ f'/processed/splited_gene_expression/{chrs}_anno.sorted.gtf') as fp:
    for line in fp:
        lines = line.strip().split()
        ensemble_id = lines[-1]
        start, end = int(lines[-3]), int(lines[-2])
        middle = ((start + end) // 2) // 100
        start_index = int(middle - (ranges//resolution//2))
        end_index = int(middle + (ranges//resolution//2))

        # three set of features

        all_features= []
        for key in features:
            data = feature_file[key]
            if start_index < 0:
                pre = [0,]* abs(start_index)
                start_index = 0
            else:
                pre = []
            if end_index > len(data):
                after = [0,] * (end_index - len(data))
            else:
                after = []
            vec = pre + data[start_index:end_index] + after
            assert len(vec) == int((ranges//resolution))
            all_features.append(vec)
        np.save(data_dir + f'/processed/feature/histone/{chrs}_{ensemble_id}.npy', np.array(all_features))
