# this file is used to bined data in feature data
import sys

resolution = int(sys.argv[1])
chrs = sys.argv[2]
types = sys.argv[3]
data_dir = sys.argv[4]

with open(data_dir + f'/processed/splited_histone_modification/{types}_{chrs}.bed', 'r') as fp:
    fpo = open(data_dir + f'/processed/splited_histone_modification/{types}_{chrs}_{resolution}.bed', 'w')

    start = 0 # zero-started
    for line in fp:
        lines = line.strip().split()
        current_start, current_end, signal = int(lines[1]), int(lines[2]), float(lines[-1])
        while (start + (resolution//2)) <= current_end:
            start += resolution
            fpo.write(f'{signal}\n')