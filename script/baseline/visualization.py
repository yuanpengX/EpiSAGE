import argparse
from pickle import NONE
import pickle as pkl
from experiment import *
from turtle import forward
from matplotlib.colors import NoNorm
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
#import networkx as nx
import torch
import dgl
from layers import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataset import GraphRegDataset, BaselineDataset, GraphRegValidDataset
from sklearn.metrics import accuracy_score
import random
from models import att_chrome
from sklearn.metrics import f1_score,roc_auc_score
torch.set_default_tensor_type(torch.DoubleTensor)



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--experiment', type=str, default='sage',
                        help='experiment name')
    parser.add_argument('--cell_line', type=str, default='mesc',
                        help='cell line data')
    parser.add_argument('--model', type=str, default='sage',
                        help='model name')
    parser.add_argument('--hidden', type=int, default=128,
                        help='model hidden num')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--layers', type=int, default=3,
                        help='n layer')
    parser.add_argument('-t','--threshold', type=float, default=3,
                        help='graph threshold')
    parser.add_argument('-f','--fold', type=int, default=1,
                        help='n layer')
    parser.add_argument('--mode', type=str, default='train',
                        help='train, test')
    parser.add_argument('--build', type=str, default='mm9',
                        help='train, test')
    parser.add_argument('--nodes', type=int, default=400,
                        help='nodes')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--head', type=int, default=4,
                        help='head attention')
    parser.add_argument('--task', type=str, default="regression",
                        help='regression,classification')
    parser.add_argument('--ag', type=str,default='mean', help='mean,lstm')
    parser.add_argument('--act', type=str,default='relu', help='relu,elu')
    parser.add_argument('--lr', type=float,default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int,default=10, help='early stop patience')
    parser.add_argument('--loss', type=str,default="mse", help='mse, poissson')
    args = parser.parse_args()
    debug = False
    args.experiment = f'{args.experiment}_{args.model}'

    experiment = Experiment(args, max_epochs=2000)
    data_dir=f'/data2/xiongyuanpeng/gene_expression/data/{args.cell_line}/processed/baseline_data/'
    experiment.load()
    if 'sage' in args.model :
        test_dataset = GraphRegValidDataset(data_dir, f'test_{args.fold}.txt',task=args.task, threshold = args.threshold, nodes  = args.nodes, pre_load = True, build=args.build)
    else:
        test_dataset = BaselineDataset(data_dir+f'test_{args.fold}.txt', task=args.task, pre_load=True, debug=debug)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 保存cnn特征，开始玩耍啦
    preds, cnn1, cnn2, fins, labels = experiment.predict(test_loader)
    pkl.dump((preds, cnn1, cnn2, fins,labels), open(f'feature_{args.experiment}.bin','wb'))
    #result_str = f'\ntest loss: {test_loss} test p {test_p}\n'
    #print(result_str)