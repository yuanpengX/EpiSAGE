from torch import norm, prelu
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import cooler
import pickle as pkl
from const import *

def parse(file):
    x  = file[0].split('/')[-1][:-4].split('_')[-1]
    return x

class GraphRegDataset(Dataset):

    def __init__(self,result_dir, file_name, task='regression', threshold = 4, nodes = 400, pre_load=True,build='mm9'):
        self.files = [line.strip().split() for line in open(result_dir + file_name)]
        self.result_dir = result_dir
        self.nodes = nodes
        self.task = task
        self.threshold = threshold
        self.build = build
        if self.build == 'mm10':
            self.max_chrom=20
        else:
            self.max_chrom=23
        self.pre_load = pre_load
        self.chrs = set()
        if pre_load:
            self.geneid2idx = {parse(file):idx for idx, file in enumerate(self.files)}
            self.data = [self.__load__(file) for file in tqdm(self.files)]
            self.median = np.median([d[-1] for d in self.data])
        self.__get_chrom_dic__()
        self.__get_graph_data__()

    def __load__(self, file):
        try:
            tss_feat, tts_feat, label = pkl.load(open(file[0],'rb'))
        except:
            print(file[0])
            exit()
        #label = label.reshape(-1)
        tss_feat[np.isnan(tss_feat)] = 0.0
        tts_feat[np.isnan(tts_feat)] = 0.0
        #label[np.isnan(label)] = 0.0
        label = np.log2(label+1)#/20.0
        return np.array(tss_feat,dtype=np.float), np.array(tts_feat), np.array(label,np.float)

    def __get_chrom_dic__(self):
        self.chrom_dict = {}
        for file in self.files:
            name = file[0].split('/')[-1][:-4]

            chrs, gene_id = name.split('_')
            self.chrs.add(chrs)
            if chrs not in self.chrom_dict:
                self.chrom_dict[chrs] = []
            self.chrom_dict[chrs].append(gene_id)

    def __get_graph_data__(self):
        self.array_dic = {}
        for i in range(1,self.max_chrom):
            chrs = f'chr{i}'
            gene_id2array_index = pkl.load(open(self.result_dir + f'{chrs}_gene_id2array_index.bin','rb'))
            interaction_array = pkl.load(open(self.result_dir + f'{chrs}_interaction_array.bin','rb'))
            self.array_dic[chrs] = (gene_id2array_index, interaction_array)

    def __len__(self):
        return (len(self.files)//self.nodes)

    def __get_data__(self, chrs, gene_ids):
        tss_feats = []
        tts_feats = []
        labels =[]
        for gene_id in gene_ids:

            tss, tts, label = self.data[self.geneid2idx[gene_id]]
            tss_feats.append(tss)
            tts_feats.append(tts)
            if self.task == 'classification':
                label = (label > self.median)*1.0
            label = [1.0,0.0] if label ==0 else [0.0,1.0]
            labels.append(label)
        tss_feats = np.array(tss_feats)
        tts_feats = np.array(tts_feats)
        #print(tss_feats.shape, tts_feats.shape)
        feats = np.concatenate((tss_feats,tts_feats),axis=-1)
        # 获取graph
        selected_notes = [self.array_dic[chrs][0][gene_id] for gene_id in gene_ids]
        mat = self.array_dic[chrs][1][selected_notes,:][:,selected_notes]
        # todo: this logic might not work
        mat = mat + np.eye(len(mat)) * self.threshold
        # end todo

        edge = np.argwhere(mat>= self.threshold)

        return np.array(feats), np.array(edge), np.array(labels,np.float)

    def __getitem__(self, index):
        if self.pre_load:
            # 先选择染色体
            chrs = random.sample(self.chrs, 1)[0]#np.random.randint(1,20)
            gene_ids = random.sample(self.chrom_dict[chrs],self.nodes)
            # 开始拼凑数据
            # 特征
            return self.__get_data__(chrs, gene_ids)
        else:
            raise NotImplementedError
            #return self.__load__(self.files[index])


class GraphRegValidDataset(GraphRegDataset):

    def __init__(self, result_dir, file_name, task='regression',threshold=4, nodes=400, pre_load=True, build='mm9'):
        super().__init__(result_dir, file_name,task, threshold, nodes, pre_load, build)
        self.__wrap_data__()

    def __wrap_data__(self):
        # 根据每个染色体的gen_ids，划分数据
        self.all_datasets = []
        for ch in self.chrs:
            all_nodes = self.chrom_dict[ch]
            for i in range(len(all_nodes)//self.nodes + 1):
                self.all_datasets.append((ch, all_nodes[i*self.nodes:(i+1)*self.nodes]))

    def __len__(self):
        return len(self.all_datasets)

    def __getitem__(self, index):
        # valid 要保证测试的时候没有重复，忠实测试所有数据
        chrs, gene_ids = self.all_datasets[index]
        return self.__get_data__(chrs, gene_ids)

class BaselineDataset(Dataset):

    def __init__(self, file_name, task='regression',pre_load = True, debug=False):

        self.task = task
        self.files = [line.strip().split() for line in open(file_name)]
        if debug:
            random.shuffle(self.files)
            self.files = self.files[:1000]

        self.pre_load = pre_load
        if pre_load:
            self.data = [self.__load__(file) for file in tqdm(self.files)]
            self.all_labels = [d[-1] for d in self.data]
            self.median = np.median(self.all_labels)
            print(sum(self.all_labels>self.median),len(self.all_labels))
    def __load__(self, file):

        tss_feat, tts_feat, label = pkl.load(open(file[0],'rb'))
        #label = label.reshape(-1)
        tss_feat[np.isnan(tss_feat)] = 0.0
        tts_feat[np.isnan(tts_feat)] = 0.0
        #label[np.isnan(label)] = 0.0
        label = np.log2(label+1)
        return np.array(tss_feat,dtype=np.float), np.array(tts_feat), np.array([label,],np.float)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.pre_load:
            tss, tts, label = self.data[index]
            if self.task =='classification':
                label = (label > self.median) * 1.0
            return tss, tts, label
        else:
            raise NotImplementedError
            #return self.__load__(self.files[index])

if __name__ == '__main__':
    cell_line = "liver"
    data="test"
    data_dir=f'/data2/xiongyuanpeng/gene_expression/data/{cell_line}/processed/baseline_data/'
    test_dataset = GraphRegValidDataset(data_dir, f'test_1.txt',task='classification', threshold = 0.4, nodes  = 700, pre_load = True, build='mm10')

    #dataset = BaselineDataset(f'/data2/xiongyuanpeng/gene_expression/data/{cell_line}/processed/baseline_data/', f'{data}_1.txt', pre_load = True)
    loader = DataLoader(test_dataset, batch_size=1)
    for feature, graph ,label  in loader:

        print(feature.shape)
        print(graph.shape)
        print(label.shape)
        break

