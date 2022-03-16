import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchsummary import summary
import matplotlib as mpl
mpl.use('Agg')
import torch.nn.functional as F


class GraphAttention(nn.Module):

    def __init__(self,
                 in_feat,
                 out_feat,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.0,
                 use_bias=False,
                 **kwargs):
        super(GraphAttention, self).__init__(**kwargs)
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')
        self.in_feat = in_feat
        self.out_feat = out_feat  # Number of output features
        self.attn_heads = attn_heads  # Number of attention heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = F.elu
        self.use_bias = use_bias

        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.out_feat * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.out_feat


        self.kernel_selfs = nn.ModuleList()
        self.kernel_neights = nn.ModuleList()
        self.atten_selfs = nn.ModuleList()
        self.atten_neights = nn.ModuleList()

        for _ in range(self.attn_heads):
            self.kernel_selfs.append(nn.Linear(in_feat, out_feat, bias = use_bias))
            self.kernel_neights.append(nn.Linear(in_feat, out_feat, bias = use_bias))
            self.atten_selfs.append(nn.Linear(out_feat, 1, bias = use_bias))
            self.atten_neights.append(nn.Linear(out_feat, 1, bias = use_bias))



    def forward(self, X, A):
        #X = inputs[0]  # Node features (B x N x F)
        #A = inputs[1]  # Adjacency matrix (B x N x N)
        outputs = []
        Att = []
        for head in range(self.attn_heads):
            kernel_self = self.kernel_selfs[head]      # W in the paper (F x F')
            kernel_neighs = self.kernel_neights[head]      #
            attention_kernel =  [self.atten_selfs[head], self.atten_neights[head] ]  # Attention kernel a in the paper (2F' x 1)

            # Compute inputs to attention network
            features_self = kernel_self(X)# (B x N x F')
            features_neighs = kernel_neighs(X)

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = attention_kernel[0](features_self) # (B x N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = attention_kernel[1](features_neighs)# (B x N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            attn_for_self_permute = attn_for_self.permute(1, 0, 2)  #    # (N x B x 1)

            attn_for_neighs_permute = attn_for_neighs.permute(1,0,2)# # (N x B x 1)
            att = attn_for_self_permute + attn_for_neighs_permute.permute(2,1,0)        # (N x B x N) via broadcasting

            att = att.permute(1,0,2)# (B x N x N)

            # Add nonlinearty
            att = F.leaky_relu(att, 0.2)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e15 * (1.0 - A) #
            att += mask

            # Apply sigmoid to get attention coefficients
            att = F.sigmoid(att)
            att_sum = torch.sum(att, dim = -1, keepdims = True)
            att = att/(1 + att_sum)
            beta_promoter = 1/(1 + att_sum)

            Att.append(att)

            # Apply dropout to features and attention coefficients
            #dropout_attn = Dropout(self.dropout_rate)(att)                    # (B x N x N)
            dropout_feat_neigh = nn.Dropout(self.dropout_rate)(features_neighs)   # (B x N x F')
            dropout_feat_self = nn.Dropout(self.dropout_rate)(features_self)      # (B x N x F')

            # Linear combination with neighbors' features
            node_features = dropout_feat_self * beta_promoter + torch.bmm(att, dropout_feat_neigh)  # (B x N x F')

            #if self.use_bias:
            #    node_features = K.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)
        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = torch.concat(outputs, dim = -1)            # (B x N x KF')
        else:
            output = torch.mean(torch.stack(outputs), dim=0)  # (B x N x F')

        output = self.activation(output)

        return output, Att

if __name__ == '__main__':
    net = GraphAttention(3, 128, 8,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = net.to(device)
    #summary(net, [(1200, 3), (1200, 1200)])
    batch_size = 2
    features = torch.tensor(np.random.randn(batch_size, 1200, 3)).to(torch.float32).cuda()
    matrix = torch.tensor(np.random.randn(batch_size, 1200, 1200)).to(torch.float32).cuda()

    output, Att = net(features, matrix)
    print(output.size())