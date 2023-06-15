import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import MessagePassing


class GCNConv_SH(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # in_channels=64 outchannels=64
        super(GCNConv_SH, self).__init__(aggr='mean')  # 对邻居节点特征进行平均操作
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # 公式2
        out = self.propagate(x=x, edge_index=edge_index)
        return self.tanh(out)

    def message(self, x_j):
        x_j = self.lin(x_j)  # m = e*T 公式1
        return x_j


class GCNConv_SS_HH(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_SS_HH, self).__init__(aggr='add')  # 对邻居节点特征进行sum操作
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index):
        # 公式10
        out = self.propagate(edge_index, x=x)
        return self.tanh(out)

    def message(self, x_j):
        x_j = self.lin(x_j)
        return x_j


class KDHR(torch.nn.Module):
    def __init__(self, ss_num, hh_num, sh_num, embedding_dim, batch_size, dropout,**kwargs):
        # ss_num=390, hh_num=811, sh_num=1201, embedding_dim=64, batchSize=32, drop=0.5
        super(KDHR, self).__init__()
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.dropout = dropout
        self.SH_embedding = torch.nn.Embedding(sh_num, embedding_dim)  # 1201*64 所有草药和症状的嵌入向量
        self.SS_embedding = torch.nn.Embedding(ss_num, embedding_dim)  # 390*64 所有症状的嵌入向量
        self.HH_embedding = torch.nn.Embedding(hh_num, embedding_dim)  # 811*64 所有药物的嵌入向量


        # S
        self.convSH_TostudyS_1 = GCNConv_SH(embedding_dim, embedding_dim)  # 64*64
        self.convSH_TostudyS_2 = GCNConv_SH(embedding_dim, embedding_dim)  # 64*64
        self.SH_mlp_1 = torch.nn.Linear(embedding_dim, 256)
        self.SH_bn_1 = torch.nn.BatchNorm1d(256)
        self.SH_tanh_1 = torch.nn.Tanh()

        # H
        self.convSH_TostudyS_1_h = GCNConv_SH(embedding_dim, embedding_dim)  # 64*64
        self.convSH_TostudyS_2_h = GCNConv_SH(embedding_dim, embedding_dim)  # 64*64
        self.SH_mlp_1_h = torch.nn.Linear(embedding_dim, 256)
        self.SH_bn_1_h = torch.nn.BatchNorm1d(256)
        self.SH_tanh_1_h = torch.nn.Tanh()

        # S-S图网络
        self.convSS = GCNConv_SS_HH(embedding_dim, 256)  # 64*256

        # H-H图网络  维度加上嵌入KG特征的维度
        self.convHH = GCNConv_SS_HH(embedding_dim+811, 256)  # (64+27)*256
        # self.convHH = GCNConv_SS_HH(embedding_dim, 256)
        # SI诱导层
        # SUM
        self.mlp = torch.nn.Linear(256, 256)

        self.chat = torch.nn.Linear(1536, 256)
        # cat
        # self.mlp = torch.nn.Linear(512, 512)
        self.SI_bn = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()

    def forward(self, x_SH, edge_index_SH, x_SS, edge_index_SS, x_HH, edge_index_HH, prescription, kgOneHot,chatid):
        '''
            x_SH 1201*1
            edge_index_SH 2*42419
            x_SS 390*1
            edge_index_SS 2*5566
            x_HH 811*1 
            edge_index_HH 2*65581
            prescription 512*390
            kgOneHot 811*811
        '''
        x_SH1 = self.SH_embedding(x_SH.long()).squeeze(1)  # 1201*1*64 ==> 1201*64
        x_SH2 = self.convSH_TostudyS_1(x_SH1.float(), edge_index_SH)  # 1201*64, 2*42419 ==> 1201*64
        x_SH6 = self.convSH_TostudyS_2(x_SH2, edge_index_SH)  # 1201*64, 2*42419 ==> 1201*64
        x_SH9 = (x_SH1 + x_SH2 + x_SH6) / 3.0
        x_SH9 = self.SH_mlp_1(x_SH9) # 1201*64 ==> 1201*256
        x_SH9 = x_SH9.view(1201, -1)
        x_SH9 = self.SH_bn_1(x_SH9)
        x_SH9 = self.SH_tanh_1(x_SH9) # 1201*256

        x_SH11 = self.SH_embedding(x_SH.long()).squeeze(1) # 1201*1*64 ==> 1201*64
        x_SH22 = self.convSH_TostudyS_1_h(x_SH11.float(), edge_index_SH) # 1201*64, 2*42419 ==> 1201*64
        x_SH66 = self.convSH_TostudyS_2_h(x_SH22, edge_index_SH) # 1201*64, 2*42419 ==> 1201*64
        x_SH99 = (x_SH11 + x_SH22 + x_SH66) / 3.0  # 1201*64
        x_SH99 = self.SH_mlp_1_h(x_SH99) # 1201*64 ==> 1201*256
        x_SH99 = x_SH99.view(1201, -1) 
        x_SH99 = self.SH_bn_1_h(x_SH99)
        x_SH99 = self.SH_tanh_1_h(x_SH99) # 1201*256

        x_ss0 = self.SS_embedding(x_SS.long()).squeeze(1) # 390*1*64 ==> 390*64
        x_ss1 = self.convSS(x_ss0.float(), edge_index_SS) # 390*64, 2*5566 ==> 390*64
        x_ss1 = x_ss1.view(390, -1)

        x_hh0 = self.HH_embedding(x_HH.long()).squeeze(1)  # 811*1*64 ==> 811*64
        # x_hh0 = x_hh0.view(-1, 64)
        x_hh0 = torch.cat((x_hh0.float(), kgOneHot.float()), dim=-1)
        x_hh1 = self.convHH(x_hh0.float(), edge_index_HH)  # H_H图中 h的嵌入
        x_hh1 = x_hh1.view(811, -1)
        
        es = x_SH9[:390] + x_ss1  
        eh = x_SH99[390:] + x_hh1  # 811*dim
        es = es.view(390, -1)

        e_synd = torch.mm(prescription, es)  # prescription * es
        preSum = prescription.sum(dim=1).view(-1, 1)
        e_synd_norm = e_synd / preSum
        e_synd_norm = self.mlp(e_synd_norm)    
        e_synd_norm = self.SI_bn(e_synd_norm)
        e_synd_norm = self.relu(e_synd_norm)  # 512*256

        if self.kwargs.get('chat'):
            chat256 = self.chat((chatid+1)/2)
            e_synd_norm = chat256*e_synd_norm

        eh = eh.view(811, -1)
        pre = torch.mm(e_synd_norm, eh.t())
        return pre
