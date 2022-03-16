from torch import norm, prelu
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import pickle as pkl
class GraphRegDataset(Dataset):

    def __init__(self, file_name, norms = True, norm_type = 'min_max', pre_load = True, debug=True):

        self.files = [line.strip().split() for line in open(file_name)]
        if debug:
            random.shuffle(self.files)
            self.files = self.files[:1000]
        self.norm = norms
        self.norm_type = norm_type

        self.pre_load = pre_load
        if pre_load:
            self.data = [self.__load__(file) for file in tqdm(self.files)]

    def __load__(self, file):

        feat1, feat2, feat3, label, matrix = pkl.load(open(file[0],'rb'))
        #print(file[1])
        matrix[np.isnan(matrix)] = 0.0
        label = label.reshape(-1)
        idx = pkl.load(open(file[1],'rb'))
        feature = np.vstack([feat1, feat2, feat3])
        feature[np.isnan(feature)] = 0.0
        label[np.isnan(label)] = 0.0
        label = np.log2(label+1)
        '''
        feature = np.load(label[0])
        matrix = np.load(label[1])
        label = float(label[2]) # label should transform
        label = np.log(label + 1) # pseudo count

        if self.norm:
            if self.norm_type == 'min_max':
                min_val = matrix.min().min()
                max_val = matrix.max().max()
                matrix = (matrix-min_val)/(max_val - min_val)
                # check nan
                matrix[np.isnan(matrix)] = 0.0
        '''
        return np.array(feature,dtype=np.float), np.array(idx), np.array(matrix,dtype=np.float), np.array(label,np.float)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.pre_load:
            return self.data[index]
        else:
            return self.__load__(self.files[index])

if __name__ == '__main__':
    dataset = GraphRegDataset('/data2/xiongyuanpeng/gene_expression/data/mesc/processed/train_1.txt', pre_load=False)
    loader = DataLoader(dataset, batch_size=1)
    for feature, idx, graph ,label  in loader:

        print(feature.shape)
        print(label.shape)
        break