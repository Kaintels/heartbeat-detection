from config.parameter import *

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, SGD, RMSprop

layers = NN_HIDDEN_LAYERS

class Net(torch.nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()

        in_nn = STATE_DIM  # 64
        out_nn = NN_FC_DENSITY  # 64
        self.activation = 'relu'

        self.l_fc = []
        for i in range(layers):
            l = torch.nn.Linear(in_nn, out_nn)
            in_nn = out_nn

            self.l_fc.append(l)
            self.add_module("l_fc_" + str(i), l)

        self.l_out_q_val = torch.nn.Linear(in_nn, ACTION_DIM)  # q-value prediction

        self.opt = RMSprop(self.parameters(), lr=OPT_LR, alpha=OPT_ALPHA)

        self.loss_f = torch.nn.MSELoss()

        self.cuda()

    def forward(self, batch):
        flow = batch

        if self.activation == 'relu':
            for l in self.l_fc:
                flow = F.relu(l(flow))

        if self.activation == 'sigmoid':
            for l in self.l_fc:
                flow = F.sigmoid(l(flow))

        a_out_q_val = self.l_out_q_val(flow)

        return a_out_q_val

    def copy_weights(self, other, rho=TARGET_RHO):
        params_other = list(other.parameters())
        params_self = list(self.parameters())

        for i in range(len(params_other)):
            val_self = params_self[i].data
            val_other = params_other[i].data
            val_new = rho * val_other + (1 - rho) * val_self

            params_self[i].data.copy_(val_new)

    def train_network(self, s, a, q_):
        s = Variable(s)
        a = Variable(a)
        q_ = Variable(q_)

        q_pred = self(s).gather(1, a)  # we have results only for performed actions

        loss_q = self.loss_f(q_pred, q_)
        # print(float(loss_q))

        self.opt.zero_grad()
        torch.nn.utils.clip_grad_norm(self.parameters(), OPT_MAX_NORM)
        loss_q.backward()
        self.opt.step()

    def set_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
