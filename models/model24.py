import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from models.register import Registers
from torch import einsum

import torch.optim as optim


class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn=False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn, self).__init__()

        #         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim).reshape(1, -1, 1, 1))

    def forward(self, x):
        # print(self.gamma.size(), x.size())
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)


@Registers.model.register
class social_stgcnn24(nn.Module):

    def __init__(self,
                 n_stgcnn=1,
                 n_txpcnn=1,
                 input_feat=2,
                 output_feat=5,
                 seq_len=8,
                 pred_seq_len=12,
                 kernel_size=3,
                 dict_len=32,
                 var1=16,
                 drop=0.1,
                 init="kmeans",
                 dict_kernel_size=3,
                 act="nn.sigmoid",
                 env="eth"):
        super().__init__()
        self.var1 = var1
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        # stgcn
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(
            st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(
                st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        # tpcnn
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_txpcnn):
            self.tpcnns.append(
                nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())
        hidden_dict = torch.ones(5, dict_len, 12)
        # hidden_dict = torch.randn(5, dict_len, 12) / math.sqrt(50)
        self.hidden_dict = nn.Parameter(hidden_dict)
        nn.init.normal_(self.hidden_dict, std=0.02)
        self.drop = nn.Dropout(drop)
        self.W1 = nn.Linear(12, var1)
        self.W2 = nn.Linear(12, var1)
        self.W3 = nn.Linear(12, var1)
        self.W4 = nn.Linear(var1, 12)
        self.ls = LayerScale(12)
        self.ls2 = LayerScale(12)
        self.Wu = nn.Linear(12, var1)
        self.Wv = nn.Linear(12, var1)
        self.Wo = nn.Linear(var1, 12)
        self.act = act_layer(act)

        # self.W4 = nn.Sequential(
        #     nn.LayerNorm(var1),
        #     nn.Linear(var1, var1),
        #     nn.LayerNorm(var1),
        #     nn.Linear(var1, var1),
        #     nn.Dropout(0.1)
        # )

    def forward(self, v, a):
        """
        First they are forward through a series of stgcn
        """
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v
        """
        A new attention module:
        a parameterized tensor \in R^{5*dict_len*12}
        v \in R^{1*12*5*num_person} -> v^T \in R^(num_person*5*12)
        q = (W1v^T) \in (num_person var1)
        k = (W2 U) \in {dict_len var1}
        A = (qk^T)/\sqrt(var) \in {num_person dict_len}
        v = W3(hidden) \in {dict_len var1}
        v = W4(Av) \in {num_person 60}
        v_new \in {60 num_person} -> {1*12*5*num_person} 
        """
        res = v
        res = res.permute(0, 2, 3, 1)
        res = res.reshape(res.shape[1], res.shape[2],
                          res.shape[3]).contiguous()
        q = self.W1(res)
        k = self.W2(self.hidden_dict)
        hidden = self.W3(self.hidden_dict)
        att = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.var1)
        att = att.softmax(dim=-1)

        res = torch.bmm(att, hidden)
        res = self.W4(res)
        res = res.permute(2, 0, 1).reshape(1, 12, 5, -1).contiguous()

        res = self.drop(res)
        res = self.ls(res)

        v = res + v
        res = v.reshape(12, 5, -1).permute(1, 2, 0)
        u1 = self.Wu(res)
        u2 = self.Wv(res)
        u2 = self.act(u2)
        res = self.Wo(u1 * u2)
        res = res.permute(2, 0, 1).reshape(1, 12, 5, -1).contiguous()
        res = self.ls2(res)

        # 之后去掉W3 W4再试试


        # end here
        # print("before output {}".format(v.size()))
        # v = self.tpcnn_ouput(res)
        v = self.tpcnn_ouput(v + res)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        # print("after output and resize {}".format(v.size()))

        return v, a


def act_layer(act):
    if act == "nn.PReLU":
        return nn.PReLU()
    elif act == "nn.ReLU":
        return nn.ReLU()
    elif act == "nn.GELU":
        return nn.GELU()
    elif act == "nn.SiLU":
        return nn.SiLU()
    elif act == "nn.Sigmoid":
        return nn.Sigmoid()
        