# train valid test split
# we held out chromosomes i and (i + 10 mod 20) + sign(i ≥ 10) for validation and chromosomes i + 1 and (i + 11 mod 20) + sign(i ≥ 9) for test, i = 1, · · · , 10, and trained on all remaining chromosomes except X and Y.
from glob import glob
import os
import sys

cell_line = sys.argv[1]

result_dir=f'/data2/xiongyuanpeng/gene_expression/data/{cell_line}/processed/baseline_data/'
def write(fold, chrs, types):
    with open(result_dir + f'{types}_{fold}.txt','w') as fp:
        for chr in chrs:
            files = glob(result_dir+f'chr{chr}_*.bin')
            # print(files)
            for file in files:
                #file2 = file[:-4]+'_idx.dat'
                fp.write(f'{file}\n')
for i in range(1,11):
    valid_chr = set([i, (i+10)%20 + (1 if i>=10 else -1)])
    test_chr = set([i+1, (i+11)%20 + (1 if i >=9 else -1 )])
    train_chr = set(range(1,20)) - test_chr - valid_chr
    write(i, train_chr, 'train')
    write(i, valid_chr, 'valid')
    write(i, test_chr, 'test')