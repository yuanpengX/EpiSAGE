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

def set_random(seed):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] =str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Experiment(nn.Module):

    model_dict = {'hybrid':HybridExpression,
    'deepchrom':DeepChrom,
    'attenchrome':att_chrome,
    'sage':GraphSAGE,
    'sage2':GraphSAGE2}

    def __write_params__(self):

        self.log.write('-----------params---------\n')
        self.log.write(f'model:{self.args.model}\n')
        self.log.write(f'hidden:{self.args.hidden}\n')
        self.log.write(f'dropout:{self.args.dropout}\n')
        self.log.write(f'threshold:{self.args.threshold}\n')
        self.log.write(f'nodes:{self.args.nodes}\n')
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
        super(Experiment, self).__init__()
        self.experiment = args.experiment
        self.args = args
        if args.mode=='train':
            self.log = open(checkpoint_dir+f'{self.experiment}.log','w')
            self.__write_params__()
        else:
            self.log = open(checkpoint_dir+f'{self.experiment}.log','a')

        self.model = self.model_dict[self.args.model](args)
        self.model.double()
        self.optimizer = Adam(self.model.parameters(), lr = self.args.lr)
        if self.args.task=='classification':
            # 强制修改损失函数类型
            self.args.loss = 'bce'
        if args.loss == 'bce':
            self.criterion = F.cross_entropy#poisson_loss#$F.mse_loss
        elif args.loss =='mse':
            self.criterion = F.mse_loss
        else:
            def corrcoef(target, pred):
                # np.corrcoef in torch from @mdo
                # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
                pred_n = pred - pred.mean()
                target_n = target - target.mean()
                pred_n = pred_n / pred_n.norm()
                target_n = target_n / target_n.norm()
                return -1*(pred_n * target_n).sum()
            self.criterion = corrcoef#nn.KLDivLoss()
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
        for tss_feat, tts_feat, label in data_loader:
            tss_feat, tts_feat, label = tss_feat.cuda(),tts_feat.cuda(), label.cuda()
            label = label.squeeze(0)
            pred,_ = self.model(tss_feat, tts_feat)
            if self.args.task =='classification':
                pred = F.softmax(pred,-1)
            #label = F.one_hot(label.permute(1,0), num_classes=2)
            loss = self.criterion(pred, label)
            loss_val.append(loss.item())
            preds.extend(pred[:,1].detach().cpu().numpy().ravel())
            labels.extend(label[:,1].cpu().numpy().ravel())
        preds = np.array(preds).reshape(-1)
        labels = np.array(labels).reshape(-1)
        if self.args.loss=='bce':
            #self.args.loss == 'bce':
            val_p = roc_auc_score(labels, preds)
        else:
            val_p = np.corrcoef(preds,labels)[0,1]

        plt.scatter(preds, labels, s= 4)
        plt.savefig(f'result/{self.experiment}/{epoch}.png')
        plt.clf()
        return np.mean(loss_val), val_p#np.corrcoef(preds, labels)[0][1]

    def predict(self, data_loader):
        self.model.eval()
        preds = []
        cnns1 = []
        cnns2 = []
        fins = []
        labels = []
        for tss_feat, tts_feat, label in data_loader:
            if self.cuda:
                tss_feat, tts_feat, label = tss_feat.cuda(),tts_feat.cuda(), label.cuda()
            #label = F.one_hot(label.permute(1,0), num_classes=2)
            label = label.squeeze(0)
            pred, cnns = self.model(tss_feat, tts_feat)
            cnn1, cnn2, fin = cnns
            if self.args.task =='classification':
                pred = F.softmax(pred,-1)
            cnns1.extend(cnn1.detach().cpu().numpy())
            preds.extend(pred.detach().cpu().numpy())
            cnns2.extend(cnn2.detach().cpu().numpy())
            fins.extend(fin.detach().cpu().numpy())
            labels.extend(label[:,1].cpu().numpy().ravel())
        preds = np.array(preds).reshape(-1)
        return preds, cnns1, cnns2,fins,  labels

    def load(self):
        self.model = torch.load(self.checkpoint_dir + self.experiment + '_best.h5')

    def train(self, train_loader, valid_loader=None):
        best_coef = 0
        patience  = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            loss_val = []
            train_pval = []
            for tss_feat, tts_feat, label in tqdm(train_loader):
                tss_feat, tts_feat, label = tss_feat.cuda(),tts_feat.cuda(), label.cuda()
                pred,_ = self.model(tss_feat, tts_feat)
                if self.args.task =='classification':
                    pred = F.softmax(pred,-1)
                label = label.squeeze(0)
                #label = F.one_hot(label.permute(1,0), num_classes=2)
                loss = self.criterion(pred, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_val.append(loss.item())

                if self.args.loss == 'bce':
                    train_p = roc_auc_score(label.cpu().numpy().ravel(),pred.detach().cpu().numpy().ravel())
                else:
                    train_p = np.corrcoef(label.cpu().numpy().ravel(),pred.detach().cpu().numpy().ravel())[0,1]
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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    set_random(0)
    args.experiment = f'{args.experiment}_{args.model}'
    os.system(f'zip -r code_bkp/{args.experiment}.zip *.py')
    import time
    #time_str = time.strftime("%Y-%m-%d-%H")
    #args.experiment = f'{args.experiment}'

    experiment = Experiment(args, max_epochs=2000)
    data_dir=f'/data2/xiongyuanpeng/gene_expression/data/{args.cell_line}/processed/baseline_data/'
    print(f"==>data_dir\n{data_dir}")
    if args.mode == 'train':
        print("==>loading train data")
        if 'sage' in args.model:
            train_dataset = GraphRegDataset(data_dir, f'train_{args.fold}.txt', task=args.task, threshold = args.threshold, nodes = args.nodes, pre_load = True, build=args.build)
        else:
            train_dataset = BaselineDataset(data_dir+f'train_{args.fold}.txt', task=args.task, pre_load=True, debug= debug)
        g = torch.Generator()
        g.manual_seed(0)
        num_workers = 10
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,)

        print("==>loading valid data")
        if 'sage' in args.model :
            valid_dataset = GraphRegValidDataset(data_dir, f'valid_{args.fold}.txt',task=args.task, threshold = args.threshold, nodes  = args.nodes, pre_load = True, build=args.build)
        else:
            valid_dataset = BaselineDataset(data_dir+f'valid_{args.fold}.txt', task=args.task, pre_load=True, debug=debug)

        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        experiment.train(train_loader, valid_loader)
    experiment.load()
    if 'sage' in args.model:
        test_dataset = GraphRegValidDataset(data_dir, f'test_{args.fold}.txt',task=args.task, threshold = args.threshold, nodes  = args.nodes, pre_load = True, build=args.build)
    else:
        test_dataset = BaselineDataset(data_dir+f'test_{args.fold}.txt', task=args.task, pre_load=True, debug=debug)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,)
    # outlier对计算损失和计算corrcoef影响很大，如何处理coeff
    test_loss, test_p = experiment.eval(test_loader,'test')
    result_str = f'\ntest loss: {test_loss} test p {test_p}\n'
    print(result_str)
    experiment.log.write(result_str)