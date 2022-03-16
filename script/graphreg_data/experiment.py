import argparse
from ast import Expression
from email.policy import default
from pickle import NONE
import time
import os
from turtle import forward
from matplotlib.colors import NoNorm
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
#import networkx as nx
import torch
import dgl
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from layers import EpiReg, experiment_dic
from torch.optim import Adam
from dataset import GraphRegDataset
from sklearn.metrics import accuracy_score
import random

torch.set_default_tensor_type(torch.DoubleTensor)

def set_random(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class GraphRegExperiment(nn.Module):

    def __write_params__(self):

        self.log.write('-----------params---------\n')
        self.log.write(f'hidden:{self.args.hidden}\n')
        self.log.write(f'model:{self.args.model}\n')
        self.log.write(f'hidden:{self.args.hidden}\n')
        self.log.write(f'dropout:{self.args.dropout}\n')
        self.log.write(f'layers:{self.args.layers}\n')
        self.log.write(f'threshold:{self.args.threshold}\n')
        self.log.write(f'fold:{self.args.fold}\n')
        self.log.write(f'aggregator_type:{self.args.ag}\n')
        self.log.write(f'activation:{self.args.act}\n')
        self.log.write(f'lr:{self.args.lr}\n')
        self.log.write('-----------params---------\n')
        self.log.flush()

    def __init__(self, args,max_epochs =2000,
        checkpoint_dir='/data2/xiongyuanpeng/gene_expression/checkpoint/'):
        super(GraphRegExperiment, self).__init__()
        self.experiment = args.experiment
        self.log = open(checkpoint_dir+f'{self.experiment}.log','w')
        self.args = args
        self.__write_params__()

        self.model = EpiReg(args)
        self.model.double()
        self.optimizer = Adam(self.model.parameters(), lr = self.args.lr)
        if args.loss == 'poisson':
            self.criterion = nn.PoissonNLLLoss(False)#poisson_loss#$F.mse_loss
        else:
            self.criterion = F.mse_loss
        self.max_epochs = max_epochs
        self.cuda = True if torch.cuda.is_available() else False

        if self.cuda:
            self.model.cuda()
        self.checkpoint_dir = checkpoint_dir
        self.patience = self.args.patience

        if not os.path.isdir(f'result/{args.experiment}'):
            os.mkdir(f'result/{args.experiment}')

    def eval(self,data_loader, epoch):
        self.model.eval()
        preds = []
        labels = []
        loss_val = []
        for feature, idx, graph, label in data_loader:
            if len(idx[0])>0:
                if self.args.model == 'sage':
                    graph = self.__matrix2graph__(graph)
                    if self.cuda:
                        feature, idx, graph, label = feature.cuda(),idx.cuda(), graph.int().to(f'cuda:0'), label.cuda()
                else:
                    feature, idx, graph, label = feature.cuda(),idx.cuda(), graph.cuda(), label.cuda()
                pred,_ = self.model(feature, graph)
                pred_idx = pred[0][idx]
                label_idx = label[0][idx]
                loss = self.criterion(pred_idx, label_idx)
                loss_val.append(loss.item())
                preds.extend(pred.detach().cpu().numpy())
                labels.extend(label.cpu().numpy())
        preds = np.array(preds).reshape(-1)
        labels = np.array(labels).reshape(-1)

        e1 = np.random.normal(0,1e-6,size=len(labels))
        e2 = np.random.normal(0,1e-6,size=len(labels))

        val_p = np.corrcoef(preds,labels)[0,1]

        plt.scatter(preds, labels)
        plt.savefig(f'result/{self.experiment}/{epoch}.png')
        plt.clf()
        return np.mean(loss_val), val_p#np.corrcoef(preds, labels)[0][1]

    def predict(self, data_loader):
        self.model.eval()
        preds = []
        for feature, graph in data_loader:
            graph = self.__matrix2graph__(graph)
            if self.cuda:
                feature, graph = feature.cuda(), graph.cuda()
            pred,_ = self.model(feature, graph)
            preds.extend(pred.detach().cpu().numpy())
        preds = np.array(preds).reshape(-1)
        return preds

    def __matrix2graph__(self, matrix):
        matrix = matrix[0].cpu().numpy()
        row, _ = matrix.shape
        #print(matrix.shape)
        edge = np.argwhere(matrix>self.args.threshold)

        g = dgl.graph((edge[:,0], edge[:,1]))
        nodes = g.num_nodes()
        if nodes < row:
            g.add_nodes(row - nodes)
        return g

    def train(self, train_loader, valid_loader=None):
        best_coef = 0
        patience  = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            loss_val = []
            train_pval = []
            for feature, idx, graph, label in tqdm(train_loader):
                if len(idx[0])>0:
                    if self.args.model == 'sage':
                        graph = self.__matrix2graph__(graph)
                        if self.cuda:
                            feature, idx, graph, label = feature.cuda(),idx.cuda(), graph.int().to(f'cuda:0'), label.cuda()
                    else:
                        feature, idx, graph, label = feature.cuda(),idx.cuda(), graph.cuda(), label.cuda()
                    # 当前模型
                    pred,_ = self.model(feature, graph)
                    pred_idx = torch.gather(pred, 1,idx)
                    label_idx = torch.gather(label, 1,idx)

                    loss = self.criterion(pred_idx, label_idx)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_val.append(loss.item())

                    #e1 = np.random.normal(0, 1e-6, size=len(label_idx.cpu().numpy().ravel()))
                    #e2 = np.random.normal(0, 1e-6, size=len(label_idx.cpu().numpy().ravel()))

                    train_p = np.corrcoef(label_idx.cpu().numpy().ravel(),pred_idx.detach().cpu().numpy().ravel())[0,1]
                    train_pval.append(train_p)
            train_loss = np.mean(loss_val)
            train_p = np.mean(train_pval)
            val_loss, val_coef = self.eval(valid_loader, epoch)
            result_str = f'Epoch {epoch}: train_loss {train_loss:.3f} train_p {train_p:.3f} val_loss {val_loss:.3f} val_coef {val_coef:.3f}'
            print(result_str)
            self.log.write(result_str+'\n')
            if val_coef > best_coef:
                best_coef = val_coef
                patience = 0
                torch.save(self.model, self.checkpoint_dir + self.experiment + '_best.h5')
            else:
                patience +=1
                if patience > self.patience:
                    self.log.write(f'best val coeff is {best_coef}')
                    return
        # if not patience
        self.log.write(f'best val coeff is {best_coef}')

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--experiment', type=str, default='sage',
                        help='experiment name')
    parser.add_argument('-m','--model', type=str, default='sage',
                        help='model name')
    parser.add_argument('-hi','--hidden', type=int, default='128',
                        help='model hidden num')
    parser.add_argument('-d','--dropout', type=float, default='0.5',
                        help='dropout rate')
    parser.add_argument('-l','--layers', type=int, default='3',
                        help='n layer')
    parser.add_argument('-t','--threshold', type=int, default='3',
                        help='graph threshold')
    parser.add_argument('-f','--fold', type=int, default='1',
                        help='n layer')
    parser.add_argument('--head', type=int, default=4,
                        help='head attention')
    parser.add_argument('--ag', type=str,default='mean', help='mean,lstm')
    parser.add_argument('--act', type=str,default='relu', help='relu,elu')
    parser.add_argument('--lr', type=float,default=1e-4, help='learning rate')
    parser.add_argument('-p','--patience', type=int,default=50, help='early stop patience')
    parser.add_argument('--loss', type=str,default="poissson", help='mse, poissson')
    args = parser.parse_args()
    debug = False
    set_random(0)
    args.experiment = f'{args.experiment}_{args.model}'
    os.system(f'zip -r code_bkp/{args.experiment}.zip *.py')
    import time
    time_str = time.strftime("%Y-%m-%d-%H")
    args.experiment = f'{args.experiment}_{time_str}'

    experiment = GraphRegExperiment(args, max_epochs=2000)
    data_dir='/data2/xiongyuanpeng/gene_expression/data/mesc/processed/'
    train_dataset = GraphRegDataset(data_dir+f'train_{args.fold}.txt', norms = False, pre_load=True, debug= debug)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    valid_dataset = GraphRegDataset(data_dir+f'valid_{args.fold}.txt', norms = False, pre_load=True, debug=debug)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    experiment.train(train_loader, valid_loader)