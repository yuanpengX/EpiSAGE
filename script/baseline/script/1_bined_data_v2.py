import sys
from tqdm import tqdm
in_name = sys.argv[1]
out_name = sys.argv[2]
resolution = int(sys.argv[3])
fpo = open(out_name,'w')

pos = resolution
current_value = 0
prechr='chr1'
with open(in_name) as fp:
    for line in tqdm(fp):
        line = line.strip().split()
        chrs = line[0]
        if not (chrs == prechr):
            # 染色体切换到了下一个
            pos = resolution
            prechr = chrs
        start = int(line[1])
        end = int(line[2])
        middle = (start+end)//2
        value = float(line[-1])
        while (middle > pos):
            fpo.write(f'{chrs}\t{current_value}\n')
            pos += resolution
            current_value = 0
        if(middle <= pos):
            current_value += value


