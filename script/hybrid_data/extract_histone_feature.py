from const import *
import sys
from tqdm import tqdm
chrs = sys.argv[1]#'chr1'
histone = sys.argv[2]
data_dir=sys.argv[3]
processed_dir = f'{data_dir}/processed/'
tts = 2000
tss = 5000
all_histones = open(processed_dir+f'splited_histone_modification/{histone}_{chrs}.bed').readlines()
print('load histone line done')
pre_start = 0
tmp = 0
with open(processed_dir+f'splited_gene_expression/{chrs}_anno.sorted.gtf') as fp:
    with open(f'{processed_dir}/feature/histone/{histone}_{chrs}.txt','w') as fp_out:
        for line in tqdm(fp):
            lines = line.strip().split()
            start = int(lines[3])
            ends = int(lines[4])
            #expression = float(lines[-1])
            # first collect all dnase peaks

            tss_array =[[]for i in range(100)]
            tts_array =[[] for i in range(40) ]
            # 下一个从前面的开始点搜索，加快速度
            #print(all_histones[pre_start])
            for signal in all_histones[pre_start:]:
                signals = signal.split()
                #print(signals)
                s = int(signals[1])
                e = int(signals[2])

                intensity = float(signals[4])
                if e <= max(start-tss, 0):
                    tmp +=1
                    continue
                if intensity < 1e-4:
                    continue
                if s >= ends + tts:
                    break
                #print(start, ends, s, e)

                s_0 = max(s - max((start - tss),0),0)
                if s_0 < 2 * tss:
                    e_0 = max(e - max((start - tss),0),0)
                    #print('TSS',s_0, e_0)
                    while s_0 < e_0:
                        try:
                            tss_array[s_0//100].append(intensity)
                        except:
                            pass
                        #print('TSS input: ',s_0//100)
                        s_0 += 100

                s_0 = max(s - max((ends - tts),0),0)
                e_0 = max(e - max((ends - tts),0),0)
                #print('TTS', s_0, e_0)
                while s_0 < e_0:
                    #print('TTS input: ',s_0//100)
                    try:
                        tts_array[s_0//100].append(intensity)
                    except:
                        pass
                    s_0+=100
            pre_start = tmp
            #print(tts_array)
            #print(tss_array)
            # calculate feature
            tss_feature = '\t'.join([str(np.mean(result)) if len(result)>0 else "0" for result in tss_array])
            tts_feature = '\t'.join([str(np.mean(result)) if len(result)>0 else "0" for result in tts_array])
            fp_out.write(f'{chrs}\t{start}\t{ends}\t{tss_feature}\t{tts_feature}\n')
            #exit()
print(f'extract feature done {chrs} {histone} done')