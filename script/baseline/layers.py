import argparse
from pickle import NONE
import time
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
from turtle import forward
from const import *
import numpy as np
#import networkx as nx
import torch
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import torch.nn.functional as F

class DeepChrom(nn.Module):

    def __init__(self, args) -> None:
        super(DeepChrom, self).__init__()
        nfeats = 5
        filtsize = 10
        width = 100
        poolsize = 5
        nstates = [50, 625, 125]

        self.conv = nn.Sequential(
            nn.Conv1d(nfeats, nstates[0], filtsize),
            nn.ReLU(),
            nn.MaxPool1d(poolsize)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(math.ceil((width-filtsize)/poolsize)*nstates[0], nstates[1]),
            nn.ReLU(),
            nn.Linear(nstates[1], nstates[2]),
            nn.ReLU(),
            nn.Linear(nstates[2],1))

    def forward(self, tss, tts):
        x = tss
        x = self.conv(x)
        x = self.fc(x)
        return  x, None

class Atten(nn.Module):

    def __init__(self, dim) -> None:
        super(Atten, self).__init__()
        self.fc1 = nn.Linear(dim, 1)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # N L F -> N *  L * 1   N  * L * F
        weight = torch.softmax(self.fc1(x), dim = 1).permute(0,2,1)

        x = torch.relu(self.fc2(x))
        out = torch.bmm(weight, x).squeeze(1) # N * F
        return out

class region(nn.Module):
    def __init__(self) -> None:
        super(region, self).__init__()
        self.conv1 = nn.Conv1d(5, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(input_size = 128, hidden_size = 128, num_layers = 1, bidirectional = True, batch_first = True)
        self.atten = Atten(256)

    def forward(self, x):
        # N 5 100
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.permute(0,2,1)
        x,_ = self.lstm(x)
        x = self.atten(x)
        return x

class HybridExpression(nn.Module):

    def __init__(self, args) -> None:
        super(HybridExpression, self).__init__()
        #
        self.tss_region = region()
        self.tts_region = region()
        self.fc = nn.Linear(512, 1)

    def forward(self, tss, tts):
        feat1 = self.tss_region(tss)
        feat2 = self.tts_region(tts)
        feat = torch.concat([feat1,feat2],dim = -1)
        predict = self.fc(feat)
        return predict,None

class GraphSAGE(nn.Module):
    def __init__(self,args):
        super(GraphSAGE, self).__init__()
        self.args = args
        self.build(args)

    def build(self, args):

        in_feats = 700#args.hidden
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
            # 顺序 bn relu drop 的顺序需要改改！！！
            nn.Linear(n_hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1)
        )
        self.flat = nn.Flatten()

    def forward(self, inputs, graph, edge_weights = None):
        h = self.flat(inputs.squeeze(0)) # L * F
        graph = graph.squeeze(0)
        g = dgl.graph((graph[:,0],graph[:,1]))
        nodes = g.num_nodes()
        # TODO:
        #if nodes < self.args.nodes:
        #    g.add_nodes(self.args.nodes - nodes)
        graph = g.int().to('cuda:0')
        for l, layer in enumerate(self.layers):
            pre_h = h
            h = layer(graph, h, edge_weights)
            h = self.bns[l](h)
            if l > 0:
                h = h+pre_h
            h = self.activation(h)
            h = self.dropout(h)

        h = self.module(h).permute(1,0) # L * 1
        return h

class GraphSAGE2(nn.Module):
    def __init__(self,args):
        super(GraphSAGE2, self).__init__()
        self.args = args
        self.build(args)

    def build(self, args):

        in_feats = 140 * 64#args.hidden
        n_hidden = args.hidden
        n_layers = args.layers
        activation= eval(f'F.{args.act}')
        dropout = args.dropout
        aggregator_type= args.ag

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.ms = nn.ModuleList()

        self.cnn = nn.Sequential(
            nn.Conv1d(5, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        for i in range(1):
            module = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            )
            self.ms.append(module)

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
            # 顺序 bn relu drop 的顺序需要改改！！！
            nn.Linear(n_hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,2)
        )
        self.flat = nn.Flatten()

    def forward(self, inputs, graph, edge_weights = None):
        h = inputs.squeeze(0) # N C F
        h = self.cnn(h) # 改变特征
        cnn1 = h
        for i in range(1):
            x = self.ms[i](h)
            h = x + h  # short cut
        # h: N C
        cnn2 = h
        h = self.flat(h) # n * F  128 *
        # 特征预提取部分
        # 图模型部分
        # 全连接部分
        graph = graph.squeeze(0)
        g = dgl.graph((graph[:,0],graph[:,1]))
        nodes = g.num_nodes()

        graph = g.int().to('cuda:0')
        for l, layer in enumerate(self.layers):
            pre_h = h
            h = layer(graph, h, edge_weights)
            h = self.bns[l](h)
            if l > 0:
                h = h+pre_h
            h = self.activation(h)
            h = self.dropout(h)
        fin =h  # N F
        h = self.module(h) # L * 1
        return h, (cnn1, cnn2, fin)
