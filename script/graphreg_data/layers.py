import argparse
from pickle import NONE
import time
from turtle import forward
import numpy as np
#import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import torch.nn.functional as F
import dgl
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
from scipy.stats import spearmanr, pearsonr
from torchsummary import summary
from gat_layer import *

act = F.relu

class convModule(nn.Module):

    def __init__(self, args) -> None:
        super(convModule,self).__init__()
        dropout_rate = args.dropout
        hidden = args.hidden
        self.act = eval(f'F.{args.act}')
        self.conv1 = nn.Conv1d(3, hidden, 25, padding=(25 - 1)//2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=(3-1)//2)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.pool2 = nn.MaxPool1d(5)
        self.drop2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv1d(hidden, hidden, 3, padding=(3-1)//2)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.pool3 = nn.MaxPool1d(5)

    def forward(self, x):
        # x.shape = N * C * L
        x = self.drop1(self.pool1(self.bn1(self.act(self.conv1(x)))))

        x = self.drop2(self.pool2(self.bn2(self.act(self.conv2(x)))))

        x = self.pool3(self.bn3(self.act(self.conv3(x))))

        return x


class GraphSAGE(nn.Module):
    def __init__(self,):
        super(GraphSAGE, self).__init__()

    def build(self, args):
        in_feats = args.hidden
        n_hidden = args.hidden
        n_layers = args.layers
        activation= eval(f'F.{args.act}')
        dropout = args.dropout
        aggregator_type= args.ag

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
            #self.bns.append(nn.BatchNorm1d(n_hidden))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type)) # activation None

        self.bns = nn.ModuleList()
        for i in range(n_layers + 1):
            self.bns.append(nn.BatchNorm1d(n_hidden))
        self.module = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(n_hidden, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

    def forward(self, inputs, graph, edge_weights = None):
        h = inputs.permute(0,2,1).squeeze(0) # N * L * F
        for l, layer in enumerate(self.layers):
            pre_h = h
            h = layer(graph, h, edge_weights)
            if l > 0:
                h = self.activation(h+pre_h)
            else:
                h = self.activation(h)
            h = self.bns[l](h)
            h = self.dropout(h)
        h = h.unsqueeze(0).permute(0,2,1) # 1 L F
        h = self.module(h)
        return h, None
        #if self.task=='regression':
        #    return h#act(h)
        #else:
        #    return act(h)

class EpiCNN(nn.Module):

    def __init__(self):
        super(EpiCNN, self).__init__()
    def build(self, args):
        self.args = args
        self.module1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(self.args.hidden, 64, 3,  padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.module = nn.ModuleList()
        for i in range(1, 1+8):
            # padding
            dilation = 2 ** i
            padding = dilation
            modes = nn.Sequential(
                nn.Dropout(0.5),
                nn.Conv1d(64, 64, 3, dilation=2**i, padding = padding),
                nn.ReLU(),
                nn.BatchNorm1d(64),
            )

            self.module.append(modes)

    def forward(self, x, A = None):
        x = self.module1(x)
        for i in range(0, 8):
            x = self.module[i](x)
        return x, None

class EpiGraph(nn.Module):

    def __init__(self):
        super(EpiGraph,self).__init__()

    def build(self, args):
        self.args = args
        self.gat = GraphAttention(self.args.hidden, int(self.args.hidden //self.args.head), self.args.head,)
        self.module = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(self.args.hidden, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
    def forward(self, x, A):
        x = x.permute(0,2,1) # N L F
        x, Attn = self.gat(x, A)
        x = x.permute(0,2,1)
        x = self.module(x)
        return x, Attn

class EpiReg(nn.Module):

    def __init__(self, args):
        super(EpiReg, self).__init__()

        # common module
        self.conv = convModule(args)
        self.middle_module = experiment_dic[args.model]
        self.args = args
        self.middle_module.build(args)
        # common module
        self.conv2 = nn.Conv1d(64, 1, 1) # N 1 L

        # our custumized preditor

        self.fc1 = nn.Linear(1200, self.args.hidden)
        self.fc2 = nn.Linear(self.args.hidden, 1)

    def forward(self, x, A = None):
        # x: 1 C L
        # A: 1200 * 1200
        #A = A.squeeze(0)
        x = self.conv(x) # N F L
        x, Attn = self.middle_module(x, A)
        #x, Attn = self.gat(x, A) # N B F
        '''
        x = F.relu(self.conv2(x)).squeeze(1) # N B

        x = self.fc2(F.relu(self.fc1(x)))
        '''
        x = self.conv2(x).squeeze(1)
        if self.args.loss == 'poissson':
            x = torch.exp(x)
        else:
            x = F.relu(x)
        return x, Attn

experiment_dic = {'graph':EpiGraph(), 'cnn': EpiCNN(),'sage':GraphSAGE()}

if __name__ == '__main__':

    import sys
    experiment = sys.argv[1]
    #experiment_dic = {'graph':EpiGraph(), 'cnn': EpiCNN(),'sage':GraphSAGE()}

    net = EpiReg(experiment_dic[experiment])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = net.to(device)

    batch_size = 2
    features = torch.tensor(np.random.randn(batch_size, 3, 60000)).to(torch.float32).cuda()
    matrix = torch.tensor(np.random.randn(batch_size, 1200, 1200)).to(torch.float32).cuda()

    output, Att = net(features, matrix)

