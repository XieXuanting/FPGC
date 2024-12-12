from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing

class FPGC(MessagePassing):
    def __init__(self, n_enc_1, n_input, dim, num, r=30):
        super(FPGC, self).__init__()
        self.fcn1 = nn.Linear(n_input, n_enc_1)
        self.fcnFM0 = nn.Linear(num, n_input)
        self.fcnFM1 = nn.Linear(n_input, n_enc_1)
        self.fcn2 = nn.Linear(dim, n_enc_1)
        self.dim = dim
        if r == 0:
            pass
        else:
            self.weight1 = Parameter(torch.FloatTensor(n_input, int(n_input/r)))
            torch.nn.init.xavier_uniform_(self.weight1)
            self.weight2 = Parameter(torch.FloatTensor(int(n_input/r), n_input))
            torch.nn.init.xavier_uniform_(self.weight2)
        self.r = r

    def forward(self, x, x0, x2, data, device):
        x1 =  self.fcn1(x)
        if self.r == 0:
            pass
        else:
            z_c = torch.mean(x0, dim = 0, keepdim=True)
            s = torch.mm(z_c,self.weight1)
            s = F.relu(s)
            s = torch.sigmoid(torch.mm(s,self.weight2))
            indices = torch.argsort(s.squeeze())
            h2 = x0*s
            h2 = h2[:,indices[-self.dim:]]
            h2 = self.fcn2(h2)
            h2 = torch.sigmoid(h2)
        x1 = h2*x1
        xFM1 =  self.fcnFM1(self.fcnFM0(x2) + x0)
        xFM1 = h2*xFM1
        h_final = F.normalize(x1, dim=1, p=2)
        h_final2 = F.normalize(xFM1, dim=1, p=2) + torch.normal(0, torch.ones_like(xFM1) * 0.01).to(device)
        return h_final, h_final2