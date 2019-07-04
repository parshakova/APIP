# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import math, gc
import string, re
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm

# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# Modification:
#   - most classes are either modified or newly created
#   - most functions are newly created
# ------------------------------------------------------------------------------

import cuda_functional as MF

class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_stype='sru',
                 concat_layers=False, padding=False, bidirectional=True, n_actions=0,func=''):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN, 'sru':0}
        self.rnn_type = rnn_stype 
        self.n_actions = n_actions
        if func:
            start_l, end_l,  func = int(func[0]), int(func[1]), func[3:]
        else:
            start_l = num_layers*2; end_l = 0
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            if 'cell' in rnn_stype:
                if 'gru' in rnn_stype:
                    self.rnns.append(CustomGRU(input_size, hidden_size,n_actions=n_actions if (i>=start_l and i<=end_l) else 0, func=func))
                elif 'lstm' in rnn_stype:
                    self.rnns.append(CustomLSTM(input_size, hidden_size))
            else:
                if self.rnn_type == 'sru':
                    self.rnns.append(MF.SRUCell(input_size, hidden_size, dropout=dropout_rate, rnn_dropout=dropout_rate, use_tanh=1,\
                                                n_actions=n_actions if (i>=start_l and i<=end_l) else 0, func=func, bidirectional=bidirectional))
                else:
                    self.rnns.append(RNN_TYPES[rnn_stype](input_size, hidden_size,
                                          num_layers=1, bidirectional=bidirectional))

    def forward(self, x, x_mask=None, c0=None, actions=None, use_a=True):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.rnn_type != 'sru':
                if self.dropout_rate > 0:
                    rnn_input = F.dropout(rnn_input,
                                          p=self.dropout_rate,
                                          training=self.training)
            # Forward
            if i == 0 and self.rnn_type=='sru':
                rnn_output = self.rnns[i](rnn_input, c0=c0, actions=actions, use_a=use_a)[0]
            else:
                if self.n_actions>0:
                    rnn_output = self.rnns[i](rnn_input, actions=actions, use_a=use_a)[0]
                else:
                    rnn_output = self.rnns[i](rnn_input)[0]

            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output.contiguous()

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1)
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_actions=0, func=""):
        super(CustomGRU, self).__init__()
        RNN_CELLTYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
        self.rnn_cf = RNN_CELLTYPES['gru'](input_size, hidden_size)
        self.rnn_cb = RNN_CELLTYPES['gru'](input_size, hidden_size)
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.n_func = func
        self.n_in = input_size

        if self.n_actions > 0:
            if func == 'g_hc':
                self.wa_f = nn.Parameter(torch.Tensor(2, self.n_actions,
                hidden_size, hidden_size
                ))
                self.wa_b = nn.Parameter(torch.Tensor(2, self.n_actions,
                hidden_size, hidden_size
                ))
                self.func = lambda a,w: torch.mul(a,F.sigmoid(torch.mm(a,w)))
            else:
                self.wa_f = nn.Parameter(torch.Tensor(self.n_actions,
                    self.n_in, self.n_in
                ))
                self.wa_b = nn.Parameter(torch.Tensor(self.n_actions,
                    self.n_in, self.n_in
                ))
                if func == 'mul_s':
                    self.func = lambda a: F.sigmoid(a) 
            self.init_weight()

    def init_weight(self):
            val_range = (3.0/self.n_in)**0.5
            self.wa_f.data.uniform_(-val_range, val_range)
            self.wa_b.data.uniform_(-val_range, val_range)

    def forward(self, inpt, actions=None):
        out_f, out_b = [], []
        seqlen = inpt.size(0)
        if self.n_actions>0 and self.n_func != 'g_hc':
            batch = inpt.size(1)
            length = inpt.size(0)
            a_oh = one_hot(actions, self.n_actions).unsqueeze(2) # [batch x n_actions x 1]
            u_f, u_b = [], []
            xt_2d = inpt.transpose(0,1).contiguous().view(batch*length, n_in)
            for a in range(self.n_actions):
                w_if = self.func(self.wa_f[a])
                u_i = xt_2d.mm(w_if).view(batch, -1)
                u_f.append(u_i)

                w_ib = self.func(self.wa_b[a])
                u_i = xt_2d.mm(w_ib).view(batch, -1)
                u_b.append(u_i)

            uf = torch.stack(u_f, 1) # [batch x actions x len*hid]
            uf = torch.mul(uf, a_oh).sum(1).view(batch, length, -1)
            inpt_f = uf.transpose(0,1).contiguous().view(length*batch, self.wa_f.size(2))

            ub = torch.stack(u_b, 1) # [batch x actions x len*hid]
            ub = torch.mul(ub, a_oh).sum(1).view(batch, length, -1)
            inpt_b = ub.transpose(0,1).contiguous().view(length*batch, self.wa_f.size(2))
        else:
            inpt_f, inpt_b = inpt, inpt

        hx = Variable(torch.zeros(inpt.size(1), self.hidden_size), requires_grad=False).cuda()
        for i in range(seqlen):
            hx = self.rnn_cf(inpt_f[i], hx)
            out_f.append(hx)

        hx = Variable(torch.zeros(inpt.size(1), self.hidden_size), requires_grad=False).cuda()
        for i in reversed(range(seqlen)):
            hx = self.rnn_cb(inpt_b[i], hx)
            out_b.append(hx)

        out_f = torch.stack(out_f, 0)
        out_b = torch.stack(out_b, 0)
        return [torch.cat([out_f, out_b], 2)]


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        RNN_CELLTYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
        self.rnn_cf = RNN_CELLTYPES['lstm'](input_size, hidden_size)
        self.rnn_cb = RNN_CELLTYPES['lstm'](input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, inpt):
        out_f, out_b = [], []
        seqlen = inpt.size(0)
        hx = Variable(torch.zeros(inpt.size(1), self.hidden_size), requires_grad=False).cuda()
        cx = Variable(torch.zeros(inpt.size(1), self.hidden_size), requires_grad=False).cuda()
        for i in range(seqlen):
            hx, cx = self.rnn_cf(inpt[i], (hx, cx))
            out_f.append(hx)

        hx = Variable(torch.zeros(inpt.size(1), self.hidden_size), requires_grad=False).cuda()
        cx = Variable(torch.zeros(inpt.size(1), self.hidden_size), requires_grad=False).cuda()
        for i in reversed(range(seqlen)):
            hx, cx = self.rnn_cb(inpt[i], (hx, cx))
            out_b.append(hx)

        out_f = torch.stack(out_f, 0)
        out_b = torch.stack(out_b, 0)
        return [torch.cat([out_f, out_b], 2)]


class MatchBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, n_actions=0,  
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,attn='act',
                 concat_layers=False, padding=False, bidirectional=False):
        super(MatchBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns_f, self.rnns_b = nn.ModuleList(), nn.ModuleList()
        if attn == 'act':
            self.nonlin = lambda x: F.softmax(x, dim=-1)
            self.attention = SeqAttentionAction(2*hidden_size, 2*hidden_size, n_actions)
        else:
            if self.training:
                self.nonlin = lambda x: F.softmax(x, dim=-1)
            else:
                self.nonlin = lambda x: x
            self.attention = SeqAttention(2*hidden_size, 2*hidden_size)
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2*hidden_size
            self.rnns_f.append(MF.SRUCell(input_size, hidden_size,
                                      dropout=dropout_rate,
                                      rnn_dropout=dropout_rate,
                                      use_tanh=1,
                                      bidirectional=False))
            self.rnns_b.append(MF.SRUCell(input_size, hidden_size,
                                      dropout=dropout_rate,
                                      rnn_dropout=dropout_rate,
                                      use_tanh=1,
                                      bidirectional=False))


    def forward(self, x, x_mask, actions=None):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        memory = x
        # Encode all layers
        outputs = [x.transpose(0, 1)]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            attn_pools_f, attn_pools_b = [0]*rnn_input.size(0), [0]*rnn_input.size(0)

            for c in range(rnn_input.size(0)):
                att = self.nonlin(self.attention(memory, rnn_input[c], x_mask, actions))
                attn_pool = torch.mul(memory, att.unsqueeze(2)).sum(1)
                attn_pools_f[c] = attn_pool.data
                del att, attn_pool 

            for c in reversed(range(rnn_input.size(0))):
                att = self.nonlin(self.attention(memory, rnn_input[c], x_mask, actions))
                attn_pool = torch.mul(memory, att.unsqueeze(2)).sum(1)
                attn_pools_b[c] = attn_pool.data
                del att, attn_pool

            inputs_f = torch.cat([rnn_input, Variable(torch.stack(attn_pools_f, 0))], 2)
            inputs_b = torch.cat([rnn_input, Variable(torch.stack(attn_pools_b, 0))], 2)
            rnn_output_f = self.rnns_f[i](inputs_f)[0]
            rnn_output_b = self.rnns_b[i](inputs_b)[0]
            del inputs_f, inputs_b
            rnn_output = torch.cat([rnn_output_f, rnn_output_b], 2)

            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output.contiguous()


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, wn=False):
        super(SeqAttnMatch, self).__init__()
        self.linear = nn.Linear(input_size, input_size)
        if wn:
            self.linear = weight_norm(self.linear, dim=None)          

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAexist(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    """
    def __init__(self, x_size,  wn=False):
        super(BilinearSeqAexist, self).__init__()
        self.w1 = nn.Linear(x_size, x_size)
        self.w2 = nn.Linear(x_size, 1)

    def forward(self, x,y):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.w1(y) 
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        alpha = F.softmax(xWy, dim=-1).unsqueeze(2)
        logits = self.w2(F.relu((x*alpha).sum(1)))
        
        probs = F.sigmoid(logits)
        return logits, probs


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, identity=False, wn=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
            if wn:
                self.linear = weight_norm(self.linear, dim=None)
                
        else:
            self.linear = None

    def forward(self, x, y, x_mask, actions):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy, dim=-1)
        else:
            # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy, dim=-1)
        return alpha


class BilinearSeqAttnMix(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, identity=False, wn=False, ):
        super(BilinearSeqAttnMix, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
            if wn:
                self.linear = weight_norm(self.linear, dim=None)
                
        else:
            self.linear = None
        self.mixing = MixingFeatures(x_size, y_size, final=True)

    def forward(self, x, y, y1, x_mask, y_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        s_d, s_q = self.mixing(x, x_mask, y, y_mask)[:2]
        #s_d = x; s_q = y1
        Wy = self.linear(y1) if self.linear is not None else y1
        xWy = (x+s_d).bmm((Wy+s_q).unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy, dim=-1)
        else:
            # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy, dim=-1)
        return alpha


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)


    def forward(self, x):
        if self.num_route_nodes != -1:
            # x: [batch * w.h.out_chan(l-1) * num_caps_(l-1)]
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsNetFin(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, n_actions):
        super(CapsNetFin, self).__init__()

        # len = 767 //400
        self.conv = nn.ModuleList()
        hid_cnn = 64
        self.cnn_layers = 4
        self.max_len = num_classes = 400 #767
        self.linear1 = nn.ModuleList() 
        self.linear2 = nn.ModuleList()
        self.v = nn.ParameterList()
        self.n_actions = n_actions
        for a in range(n_actions):
            self.linear1.append(nn.Linear(y_size, x_size//3))
            self.linear2.append(nn.Linear(x_size, x_size//3))
            self.v.append(nn.Parameter(torch.Tensor(x_size//3)))

        self.digit_capsules1 = CapsuleLayer(num_capsules=num_classes, num_route_nodes=x_size//3, in_channels=n_actions,
                                           out_channels=20)
        self.digit_capsules2 = CapsuleLayer(num_capsules=num_classes, num_route_nodes=x_size//3, in_channels=n_actions,
                                           out_channels=20)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.v[0].size(0))
        for a in range(self.n_actions):
            self.v[a].data.uniform_(-stdv, stdv)


    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x, y, x_mask, actions):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        actions = batch * n_actions
        """
        init_len = x.size(1)
        x = torch.cat((x[:,-1,:x.size(2)//2], x[:,0,x.size(2)//2:]), 1)
        outputs = [(torch.mul(self.v[a], F.tanh(self.linear1[a](y)+self.linear2[a](x)))).unsqueeze(-1) for a in range(self.n_actions)]
        outputs = torch.cat(outputs, dim=-1)
        x = self.squash(outputs) # [batch, x_size, n_actions]


        start = self.digit_capsules1(x).squeeze(2).squeeze(2).transpose(0, 1)
        end =   self.digit_capsules2(x).squeeze(2).squeeze(2).transpose(0, 1)

        for i, inp in enumerate([start, end]): 
            classes = (inp ** 2).sum(dim=-1) ** 0.5
            minel = torch.min(classes.data)
            if minel > 0:
                minel = minel*1e-2
            else:
                minel = minel*1e-2+minel
            if init_len < self.max_len:
                classes = classes[:,:init_len]
            else:
                # problem when pad from the beginning with infinity, NLL has nonzero probability
                classes = F.pad(classes, (init_len-self.max_len,0), "constant", minel)

            classes.data.masked_fill_(x_mask.data, -float('inf'))
            if self.training:
                res = F.log_softmax(classes, dim=-1)
            else:
                res = F.softmax(classes, dim=-1)
            if i ==0:
                alpha = res
            elif i == 1:
                beta = res

        return alpha[:, :init_len], beta[:, :init_len]


class CapsNetFin_init(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size):
        super(CapsNetFin_init, self).__init__()

        # len = 767 //400
        self.conv = nn.ModuleList()
        hid_cnn = 64
        self.max_len = num_classes = 400 #767
        self.cnn_layers = 4
        self.linear = nn.Linear(y_size, x_size)
        for i in range(self.cnn_layers):
            chan_in = hid_cnn
            chan_out = hid_cnn
            kern = 9
            pad = kern//2
            step = 2
            if i == 0:
                step=1
                chan_in = x_size
            self.conv.append(nn.Conv1d(in_channels=chan_in, out_channels=chan_out, kernel_size=kern, stride=step, padding=pad))

        # len = 192 //50
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=hid_cnn, out_channels=16,
                                             kernel_size=9, stride=2)
        # len = 92 //21
        self.digit_capsules1 = CapsuleLayer(num_capsules=num_classes, num_route_nodes=16 * 21, in_channels=8,
                                           out_channels=16)
        self.digit_capsules2 = CapsuleLayer(num_capsules=num_classes, num_route_nodes=16 * 21, in_channels=8,
                                           out_channels=16)



    def forward(self, x, y, x_mask, actions):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        actions = batch * n_actions
        """
        Wy = self.linear(y)
        init_len = x.size(1)
        xWy = x + Wy.unsqueeze(1)
        if init_len < self.max_len:
            xWy = F.pad(xWy, (0,0,0,self.max_len - init_len,0,0), "constant", 0)
        else:
            xWy = xWy[:,-self.max_len:,:]
        
        #  xWy [batch, seq, hid]
        # [emb, hid, k] >> [1, emb, hid, k] >> [k, emb, hid, 1] >> [1, k, emb, hid]
        out = xWy.transpose(1,2).contiguous()
        for i in range(self.cnn_layers):
            out = F.relu(self.conv[i](out))

        x = out
        x = self.primary_capsules(x)
        start = self.digit_capsules1(x).squeeze(2).squeeze(2).transpose(0, 1)
        end =   self.digit_capsules2(x).squeeze(2).squeeze(2).transpose(0, 1)

        for i, inp in enumerate([start, end]):            
            classes = (inp ** 2).sum(dim=-1) ** 0.5
            if init_len < self.max_len:
                classes = classes[:,:init_len]
            else:
                classes = F.pad(classes, (init_len-self.max_len,0), "constant", -float('inf'))

            classes.data.masked_fill_(x_mask.data, -float('inf'))
            if self.training:
                res = F.log_softmax(classes, dim=-1)
            else:
                res = F.softmax(classes, dim=-1)

            if i ==0:
                alpha = res
            elif i == 1:
                beta = res

        return alpha[:, :init_len], beta[:, :init_len]


class BilinearSeqAttnAction1(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, n_actions, identity=False, wn=False, func="h"):
        super(BilinearSeqAttnAction1, self).__init__()
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.weight = nn.Parameter(torch.Tensor(y_size, x_size))
        self.bias = nn.Parameter(torch.Tensor(x_size))
        self.n_actions = n_actions
        self.n_func = func
        if func == 'h':
            self.wa_h = nn.Parameter(torch.Tensor(self.n_actions, y_size, 1))
            def attn(w1, wh):
                # w1 [emb x hid]
                # wh [emb x 1]
                a2 = torch.mul(w1, wh)
                score2 = F.softmax(a2.sum(0), dim=-1).unsqueeze(0) # [1 x hid]
                wr = torch.mul(w1, score2)
                return wr
            self.func = lambda a,b : attn(a,b) 
        elif func == 'eh':
            self.wa_h = nn.Parameter(torch.Tensor(self.n_actions, y_size, 1))
            self.wa_e = nn.Parameter(torch.Tensor(self.n_actions, 1, x_size))
            def attn(w1, wh, we):
                # w1 [emb x 3*(2hid)]
                # wh [emb x 1]
                # we [1 x 3*(2hid)]
                a1 = torch.mul(w1, we)
                score1 = F.softmax(a1.sum(1), dim=-1).unsqueeze(1) # [emb x 1]
                a2 = torch.mul(w1, wh)
                score2 = F.softmax(a2.sum(0), dim=-1).unsqueeze(0) # [1 x 3*(2hid)]
                wr = torch.mul(w1, score1)
                wr = torch.mul(wr, score2)
                return wr
            self.func = lambda a,b,c : attn(a,b,c)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()
        if self.n_func == 'h':
            stdv = 1. / math.sqrt(self.wa_h.size(1))
            self.wa_h.data.uniform_(-stdv, stdv)
        else:
            self.wa_e.data.uniform_(-stdv, stdv)
            stdv = 1. / math.sqrt(self.wa_h.size(1))
            self.wa_h.data.uniform_(-stdv, stdv)

    def forward(self, x, y, x_mask, actions):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        actions = batch * n_actions
        """
        a_oh = one_hot(actions, self.n_actions).unsqueeze(2) # [batch x n_actions x 1]
        u = []
        for a in range(self.n_actions):
            if self.n_func == 'h':
                w_i = self.func(self.weight, self.wa_h[a])
            elif self.n_func == 'eh':
                w_i = self.func(self.weight, self.wa_h[a], self.wa_e[a])
            u_i = y.mm(w_i)
            u.append(u_i)
        u = torch.stack(u, 1) # [batch x actions x hid]
        Wy = torch.mul(u, a_oh).sum(1) + self.bias

        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy, dim=-1)
        else:
            # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy, dim=-1)
        return alpha


class BilinearSeqAttnAction(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, n_actions, identity=False, wn=False, func='kconv5'):
        super(BilinearSeqAttnAction, self).__init__()
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.weight = nn.Parameter(torch.Tensor(y_size, x_size))
        self.bias = nn.Parameter(torch.Tensor(x_size))
        
        self.w_conv = nn.ModuleList()
        self.n_actions = n_actions
        self.cnn_layers = int(func[5:].split('_')[0])
        hid_cnn = 64
        for i in range(self.cnn_layers):
            chan_in = hid_cnn
            chan_out = hid_cnn
            kern = 3
            pad = 1
            if i == 0:
                chan_in = 1
            elif i == self.cnn_layers-1:
                #kern = 1
                #pad = 0
                chan_out = 1
            a_conv = nn.ModuleList()
            for a in range(self.n_actions):
                a_conv.append(nn.Conv2d(chan_in, chan_out, kern, stride=1, padding=pad))

            self.w_conv.append(a_conv)

        def conv_forw(a):
            # w1 [emb x 3*(2hid)]
            out = self.weight.unsqueeze(0).unsqueeze(0)
            for i in range(self.cnn_layers):
                if i != self.cnn_layers-1:
                    out = F.relu(self.w_conv[i][a](out))
                else:
                    out = self.w_conv[i][a](out)
            out = out.squeeze()
            return out

        self.func = lambda a: conv_forw(a)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, x, y, x_mask, actions):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        actions = batch * n_actions
        """
        a_oh = one_hot(actions, self.n_actions).unsqueeze(2) # [batch x n_actions x 1]
        u = []
        for a in range(self.n_actions):
            w_i = self.func(a)
            u_i = y.mm(w_i)
            u.append(u_i)
        u = torch.stack(u, 1) # [batch x actions x hid]
        Wy = torch.mul(u, a_oh).sum(1)

        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy, dim=-1)
        else:
            # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy, dim=-1)
        return alpha


class BilinearSeqAttnAction3(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, n_actions, identity=False, wn=False, func='mul_s'):
        super(BilinearSeqAttnAction3, self).__init__()
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.weight = nn.Parameter(torch.Tensor(y_size, x_size))
        self.bias = nn.Parameter(torch.Tensor(x_size))
        self.wa = nn.Parameter(torch.Tensor(n_actions,y_size, x_size))
        self.ba = nn.Parameter(torch.Tensor(n_actions, x_size))
        self.n_actions = n_actions
        if func == 'mul':
            self.func = lambda a,b: torch.mul(a,b)
        elif func == 'mul_s':
            self.func = lambda a,b: torch.mul(a,F.sigmoid(b))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()
        self.ba.data.zero_()
        self.wa.data.uniform_(-stdv, stdv)

    def forward(self, x, y, x_mask, actions):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        actions = batch * n_actions
        """
        a_oh = one_hot(actions, self.n_actions).unsqueeze(2) # [batch x n_actions x 1]
        u = []
        for a in range(self.n_actions):
            w_i = self.func(self.weight, self.wa[a])
            u_i = y.mm(w_i)
            u.append(u_i)
        u = torch.stack(u, 1) # [batch x actions x hid]
        b = self.func(self.bias, torch.mm(a_oh.squeeze(2), self.ba))
        Wy = torch.mul(u, a_oh).sum(1) + b

        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy, dim=-1)
        else:
            # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy, dim=-1)
        return alpha


class BilinearSeqAttnAction2(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, n_actions, identity=False, wn=False):
        super(BilinearSeqAttnAction2, self).__init__()
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.weight = nn.Parameter(torch.Tensor(n_actions,y_size, x_size))
        self.bias = nn.Parameter(torch.Tensor(n_actions,x_size))
        self.n_actions = n_actions
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, y, x_mask, actions):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        actions = batch * n_actions
        """
        a_onehot = one_hot(actions, self.n_actions)
        w = torch.mm(a_onehot, self.weight.view(self.n_actions, -1)).view(x.size(0), self.weight.size(1), self.weight.size(2))
        b = torch.mm(a_onehot, self.bias)
        Wy = torch.bmm(y.unsqueeze(1), w).squeeze(1) + b
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy, dim=-1)
        else:
            # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy, dim=-1)
        return alpha


class PointerNetworkAction(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, n_actions, wn=False, opt=None):
        super(PointerNetworkAction, self).__init__()
        self.attention = SeqAttentionAction(
                x_size,
                y_size, opt['n_actions'], drop_r=opt['dropout_rnn'])
        self.n_actions = n_actions
        self.rnn_cell = MF.SRUCell(
            x_size, y_size,
            bidirectional=False,dropout=opt['dropout_rnn'],rnn_dropout=opt['dropout_rnn'],
            use_tanh=1)

    def forward(self, x, x_mask, c0, actions):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        actions = batch * n_actions
        """
        s_logits = self.attention(x, c0, x_mask, actions)
        s_probs = F.softmax(s_logits, dim=-1)
        attn_pool = (x*s_probs.unsqueeze(2)).sum(1)
        state = self.rnn_cell(attn_pool, c0=c0)[1]
        e_logits = self.attention(x, state, x_mask, actions)

        if self.training:
            nonlin = lambda x: F.log_softmax(x, dim=-1)
        else:
            nonlin = lambda x: F.softmax(x, dim=-1)
        return nonlin(s_logits), nonlin(e_logits)


class PointerNetwork(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, wn=False, opt=None):
        super(PointerNetwork, self).__init__()
        self.attention = SeqAttention(
                x_size,
                y_size, wn=wn, drop_r=opt['dropout_rnn'])
        
        self.rnn_cell = MF.SRUCell(
            x_size, y_size,
            bidirectional=False,dropout=opt['dropout_rnn'],rnn_dropout=opt['dropout_rnn'],
            use_tanh=1)

    def forward(self, x, x_mask, c0, actions):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        s_logits = self.attention(x, c0, x_mask, log=True)
        s_probs = F.softmax(s_logits, dim=-1)
        attn_pool = (x*s_probs.unsqueeze(2)).sum(1)
        state = self.rnn_cell(attn_pool, c0=c0)[1]
        e_logits = self.attention(x, state, x_mask)

        if self.training:
            nonlin = lambda x: F.log_softmax(x, dim=-1)
        else:
            nonlin = lambda x: F.softmax(x, dim=-1)
        return nonlin(s_logits), nonlin(e_logits)


class SeqAttentionAction(nn.Module):
    """attention between a sequence and a tensor:
    * o_i = softmax(v*tanh(W1x_i+W2y)) for x_i in X.
    """
    def __init__(self, x_size, y_size, n_actions, wn=False, drop_r=0.0):
        super(SeqAttentionAction, self).__init__()
        self.n_actions = n_actions
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.w1 = nn.Parameter(torch.Tensor(n_actions,x_size, x_size//4))
        self.b1 = nn.Parameter(torch.Tensor(n_actions,x_size//4))
        self.w2 = nn.Parameter(torch.Tensor(n_actions,y_size, x_size//4))
        self.b2 = nn.Parameter(torch.Tensor(n_actions,x_size//4))
        self.v = nn.Parameter(torch.Tensor(n_actions,x_size//4))
        self.reset_parameters()
        if drop_r>0:
            self.dropout = nn.Dropout(drop_r)
        self.drop_r = drop_r

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w1.size(2))
        self.w1.data.uniform_(-stdv, stdv)
        self.b1.data.zero_()
        self.w2.data.uniform_(-stdv, stdv)
        self.b2.data.zero_()
        self.v.data.uniform_(-stdv, stdv)

    def get_action_parameters(self, a_onehot, x_size):
        w1 = torch.mm(a_onehot, self.w1.view(self.n_actions, -1)).view(x_size[0], self.w1.size(1), self.w1.size(2))
        w1 = w1.unsqueeze(1).expand(x_size[0], x_size[1], w1.size(1), w1.size(2))
        w1 = w1.contiguous().view(-1,w1.size(2), w1.size(3))

        w2 = torch.mm(a_onehot, self.w2.view(self.n_actions, -1)).view(x_size[0], self.w2.size(1), self.w2.size(2))
        w2 = w2.unsqueeze(1).expand(x_size[0], x_size[1], w2.size(1), w2.size(2))
        w2 = w2.contiguous().view(-1,w2.size(2), w2.size(3))

        b1 = torch.mm(a_onehot, self.b1).unsqueeze(1).expand(x_size[0], x_size[1], self.b1.size(1)).contiguous().view(-1, self.b1.size(1))
        b2 = torch.mm(a_onehot, self.b2).unsqueeze(1).expand(x_size[0], x_size[1], self.b2.size(1)).contiguous().view(-1, self.b2.size(1))
        v = torch.mm(a_onehot, self.v).unsqueeze(1).expand(x_size[0], x_size[1], self.v.size(1)).contiguous().view(-1, self.v.size(1))
        return w1, w2, b1, b2, v

    def forward(self, x, y, x_mask, actions):
        """
        x = batch * len * hdim
        y = batch * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, 1, x.size(-1))
        y_flat = y.unsqueeze(1).expand(y.size(0), x.size(1), y.size(1)).contiguous().view(-1, 1, y.size(-1))
        a_onehot = one_hot(actions, self.n_actions)
        w1, w2, b1, b2, v = self.get_action_parameters(a_onehot, [x.size(0), x.size(1), x.size(2)])
        x_t = torch.bmm(x_flat, w1).squeeze(1) + b1
        y_t = torch.bmm(y_flat, w2).squeeze(1) + b2
        
        inpt = F.tanh(x_t+y_t)
        if self.drop_r>0:
            inpt = self.dropout(inpt)
        inpt = torch.bmm(inpt.unsqueeze(1), v.unsqueeze(2))
        scores = inpt.view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        del w1,w2,b1,b2,v,x_flat,y_flat,inpt
        return scores


class CriticLinear(nn.Module):
    def __init__(self, x_size, y_size, identity=False, num_layers=2, wn=False, nl=4):
        super(CriticLinear, self).__init__()
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.w1 = self.wn(nn.Linear(x_size, y_size))
        if num_layers == 3:
            self.w2 = self.wn(nn.Linear(y_size, y_size))
        if nl == 3:
            self.w3 = self.wn(nn.Linear(y_size, 1))
        elif nl == 4:
            self.w3 = self.wn(nn.Linear(y_size, y_size))
            self.w4 = self.wn(nn.Linear(y_size, 1))
        self.nl = nl
        self.num_layers = num_layers

    def forward(self, x):
        c1 = self.w1(x) 
        if self.num_layers == 3:
            c1 = self.w2(F.relu(c1))
        if self.nl == 4:
            c2 = self.w4(F.relu(self.w3(F.relu(c1)))) 
        else:
            c2 = (self.w3(F.relu(c1)))
        return c2.squeeze(1)


class PolicyLatent(nn.Module):
    def __init__(self, x_size, y_size, n_actions, num_layers=2, identity=False, wn=False, add=1, nl=5):
        super(PolicyLatent, self).__init__()
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.add = add
        self.n_actions = n_actions
        if add == 3:
            self.w1a = self.wn(nn.Linear(x_size//3, y_size))
            self.w1b = self.wn(nn.Linear(x_size//3, y_size))
            self.w1c = self.wn(nn.Linear(x_size//3, y_size))
        elif add == 2:
            self.w1a = self.wn(nn.Linear(x_size//2, y_size))
            self.w1b = self.wn(nn.Linear(x_size//2, y_size))
        else:
            self.w1 = self.wn(nn.Linear(x_size, y_size))
        self.num_layers = num_layers
        if num_layers == 3:
            self.w2 = self.wn(nn.Linear(y_size, y_size))
        self.nl = nl
        if nl == 3:
            self.w3 = self.wn(nn.Linear(y_size, n_actions)) 
        else:
            self.w3 = self.wn(nn.Linear(y_size, y_size)) 
            if nl == 4:
                self.w4 = self.wn(nn.Linear(y_size, n_actions))
            elif nl ==5:
                self.w4 = self.wn(nn.Linear(y_size, y_size))
                self.w5 = self.wn(nn.Linear(y_size, n_actions))

    def forward(self, x):
        if self.add==3:
            x = self.w1a(x[:,:int(x.size(-1)//3)]) +  self.w1b(x[:,int(x.size(-1)//3):2*int(x.size(-1)//3)]) \
                                                    +  self.w1c(x[:,2*int(x.size(-1)//3):])
        elif self.add==2:
            x = self.w1a(x[:,:int(x.size(-1)/2)]) +  self.w1b(x[:,int(x.size(-1)/2):])
        else: 
            x = self.w1(x) 
        if self.num_layers == 3:
            x = self.w2(F.relu(x))
        if self.nl == 3:
            logits = self.w3(F.relu(x))
        else:
            x = self.w3(F.relu(x))
            if self.nl ==4: 
                logits = self.w4(F.relu(x))
            else:
                logits = self.w5(F.relu(self.w4(F.relu(x))))
        if self.n_actions > 1: 
            probs = F.softmax(logits, dim=-1)
        elif self.n_actions == 1: 
            probs = F.sigmoid(logits)
        return logits, probs


class ControlVector(nn.Module):
    def __init__(self, x_size, gate, n_actions, identity=False, wn=False, drop_r=0.0):
        super(ControlVector, self).__init__()
        d_factor = int(gate.split('_')[-1])
        self.gate = gate; self.n_actions = n_actions
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        if 'fc_cat' in self.gate:
            self.g = self.wn(nn.Linear(n_actions, x_size*d_factor//2))
            self.a1 = self.wn(nn.Linear(x_size, x_size*d_factor//2))
            self.a2 = self.wn(nn.Linear(n_actions, x_size*d_factor//2))
        elif 'fc_add' in self.gate:
            self.g = self.wn(nn.Linear(n_actions, x_size*d_factor))
            self.a1 = self.wn(nn.Linear(x_size, x_size*d_factor))
            self.a2 = self.wn(nn.Linear(n_actions, x_size*d_factor))
        elif 'tanh' in self.gate:
            self.w1 = nn.Parameter(torch.Tensor(n_actions,x_size, x_size*d_factor))
            self.b1 = nn.Parameter(torch.Tensor(n_actions,x_size*d_factor))
            self.w2 = nn.Parameter(torch.Tensor(n_actions,x_size, x_size*d_factor))
            self.b2 = nn.Parameter(torch.Tensor(n_actions,x_size*d_factor))
            self.reset_parameters()
        if drop_r>0:
            self.dropout = nn.Dropout(drop_r)
        self.drop_r = drop_r

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w1.size(2))
        self.w1.data.uniform_(-stdv, stdv)
        self.b1.data.zero_()
        self.w2.data.uniform_(-stdv, stdv)
        self.b2.data.zero_()

    def forward(self, x, actions): # x = batch * nhid
        a_onehot = one_hot(actions, self.n_actions)
        if 'fc_cat' in self.gate:
            gate = F.sigmoid(self.g(a_onehot)) 
            a1 = self.a1(x) 
            a2 = self.a2(a_onehot)
            res = F.tanh(torch.cat((a2, a1*gate), 1))
        elif 'fc_add' in self.gate:
            gate = F.sigmoid(self.g(a_onehot)) 
            a1 = self.a1(x) 
            a2 = self.a2(a_onehot)
            res = F.tanh(a2 + a1*gate)
        elif 'tanh' in self.gate:
            w1 = torch.mm(a_onehot, self.w1.view(self.n_actions, -1)).view(x.size(0), self.w1.size(1), self.w1.size(2))
            b1 = torch.mm(a_onehot, self.b1)
            w2 = torch.mm(a_onehot, self.w2.view(self.n_actions, -1)).view(x.size(0), self.w2.size(1), self.w2.size(2))
            b2 = torch.mm(a_onehot, self.b2)
            x_hat = F.tanh(torch.bmm(x.unsqueeze(1), w1).squeeze(1) + b1)
            gate = F.sigmoid(torch.bmm(x.unsqueeze(1), w2).squeeze(1) + b2)
            res = x_hat * gate        
        if self.drop_r>0.0:
            return self.dropout(res.contiguous())
        else:
            return res.contiguous()


class MixingFeatures(nn.Module):
    """ mixing features of sequence and a single vector
        or
        mixing features of two sequences
    """
    def __init__(self, x_size, y_size, wn=False, latent=False, final=False):
        super(MixingFeatures, self).__init__()
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.w1 = self.wn(nn.Linear(y_size, x_size))
        self.latent = latent; self.final = final
        if not latent and not final:
            self.w2 = self.wn(nn.Linear(x_size, x_size//2))
            self.w2b = self.wn(nn.Linear(x_size, x_size//2))
            self.w3 = self.wn(nn.Linear(x_size//2, 1))

    def forward(self, x, x_mask, y, y_mask):
        """
        x = batch * dlen * hdim
        x_mask = batch * dlen
        y = batch * qlen * hdim
        """
        y_n = F.tanh(self.w1(y.view(-1, y.size(-1)))).view(y.size(0), y.size(1), x.size(2))
        y_p = y_n.permute(0, 2, 1)
        A = torch.bmm(x, y_p) # batch * dlen * qlen
        A.data.masked_fill_(x_mask.data.unsqueeze(2), -float('inf'))
        A.data.masked_fill_(y_mask.data.unsqueeze(1), -float('inf'))
        # which context words are most relevant to one of query words
        m_alpha_d = F.softmax(torch.max(A, 2)[0], dim=-1)
        m_d = torch.mul(x, m_alpha_d.unsqueeze(2)).sum(1)
        if self.final:
            #s_q = torch.bmm(x.permute(0,2,1), F.softmax(A, dim=1)).permute(0,2,1) # b * qlen * hdim
            p_d = F.softmax(A, dim=2)
            mask_d = (p_d != p_d).byte()
            p_d.data.masked_fill_(mask_d.data, 0.0)
            s_d = torch.bmm(p_d, y_n) # b * dlen * hdim
            return s_d, m_d, 0
        # which question words are most relevant to one of context words
        m_alpha_q = F.softmax(torch.max(A, 1)[0], dim=-1)
        m_q = torch.mul(y_n, m_alpha_q.unsqueeze(2)).sum(1)
        ae_prob = None
        if not self.latent:
            ae_prob = F.sigmoid(self.w3(F.relu(self.w2b(m_d) + self.w2(m_q))))
        return m_d, m_q, ae_prob


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size, wn=False):
        super(LinearSeqAttn, self).__init__()
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.linear = self.wn(nn.Linear(input_size, 1))

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha


class LinearSeqAttnAction(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size,n_actions, wn=False, drop_r=0.0):
        super(LinearSeqAttnAction, self).__init__()
        self.n_actions = n_actions
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.weight = nn.Parameter(torch.Tensor(n_actions,input_size, 1))
        self.bias = nn.Parameter(torch.Tensor(n_actions,1))
        self.n_actions = n_actions
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, x, x_mask, actions):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        a_onehot = one_hot(actions, self.n_actions)
        w = torch.mm(a_onehot, self.weight.view(self.n_actions, -1)).view(x.size(0), self.weight.size(1), self.weight.size(2))
        b = torch.mm(a_onehot, self.bias).unsqueeze(1).expand(x.size(0), x.size(1), self.bias.size(1)).contiguous().view(x_flat.size(0), self.bias.size(1))
        w = w.unsqueeze(1).expand(x.size(0), x.size(1), w.size(1), w.size(2)).contiguous().view(x_flat.size(0),w.size(1), w.size(2))
        scores = (torch.bmm(x_flat.unsqueeze(1), w).squeeze(1) + b).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha


class LinearSeqAttnAction_ad1(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size,n_actions, wn=False, drop_r=0.0):
        super(LinearSeqAttnAction_ad1, self).__init__()
        self.n_actions = n_actions
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.v = nn.Parameter(torch.Tensor(n_actions,input_size))
        self.w = self.wn(nn.Linear(input_size, input_size))
        self.n_actions = n_actions
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.v.size(1))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, x, x_mask, actions):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        a_onehot = one_hot(actions, self.n_actions)
        v = torch.mm(a_onehot, self.v).unsqueeze(1).expand(x.size(0), x.size(1), self.v.size(1)).contiguous().view(x_flat.size(0), self.v.size(1))
        
        wx = self.w(x_flat)
        scores = torch.mul(wx, F.sigmoid(v)).sum(1).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha


class LinearSeqAttnAction2(nn.Module):
    """Self attention over a sequence:
    * o_i = W_2*relu(W1 x_i) for x_i in X.
    """
    def __init__(self, input_size,n_actions, wn=False, drop_r=0.0):
        super(LinearSeqAttnAction2, self).__init__()
        self.n_actions = n_actions
        self.w1 = nn.Parameter(torch.Tensor(n_actions,input_size, input_size//2))
        self.b1 = nn.Parameter(torch.Tensor(n_actions,input_size//2))
        self.w2 = nn.Parameter(torch.Tensor(n_actions,input_size//2, 1))
        self.b2 = nn.Parameter(torch.Tensor(n_actions,1))
        self.n_actions = n_actions
        self.reset_parameters()
        if drop_r>0:
            self.dropout = nn.Dropout(drop_r)
        self.drop_r = drop_r

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w1.size(2))
        self.w1.data.uniform_(-stdv, stdv)
        self.b1.data.uniform_(-stdv, stdv)
        self.w2.data.uniform_(-stdv, stdv)
        self.b2.data.uniform_(-stdv, stdv)

    def forward(self, x, x_mask, actions):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        a_onehot = one_hot(actions, self.n_actions)
        w1 = torch.mm(a_onehot, self.w1.view(self.n_actions, -1)).view(x.size(0), self.w1.size(1), self.w1.size(2))
        b1 = torch.mm(a_onehot, self.b1).unsqueeze(1).expand(x.size(0), x.size(1), self.b1.size(1)).contiguous().view(x_flat.size(0), self.b1.size(1))
        w1 = w1.unsqueeze(1).expand(x.size(0), x.size(1), w1.size(1), w1.size(2)).contiguous().view(x_flat.size(0),w1.size(1), w1.size(2))
        w2 = torch.mm(a_onehot, self.w2.view(self.n_actions, -1)).view(x.size(0), self.w2.size(1), self.w2.size(2))
        b2 = torch.mm(a_onehot, self.b2).unsqueeze(1).expand(x.size(0), x.size(1), self.b2.size(1)).contiguous().view(x_flat.size(0), self.b2.size(1))
        w2 = w2.unsqueeze(1).expand(x.size(0), x.size(1), w2.size(1), w2.size(2)).contiguous().view(x_flat.size(0),w2.size(1), w2.size(2))

        scores = F.relu(torch.bmm(x_flat.unsqueeze(1), w1) + b1.unsqueeze(1))
        if self.drop_r>0:
            scores=self.dropout(scores)
        scores = (torch.bmm(scores, w2).squeeze(1) + b2).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        return alpha


class SeqAttention(nn.Module):
    """attention between a sequence and a tensor:
    * o_i = softmax(v*tanh(W1x_i+W2y)) for x_i in X.
    """
    def __init__(self, x_size, y_size, wn=False, drop_r=0.0):
        super(SeqAttention, self).__init__()
        if wn:
            self.wn = lambda x: weight_norm(x, dim=None)
        else:
            self.wn = lambda x: x
        self.w1 = self.wn(nn.Linear(x_size, x_size))
        self.w2 = self.wn(nn.Linear(y_size, x_size))
        self.v = self.wn(nn.Linear(x_size, 1))
        if drop_r>0:
            self.dropout = nn.Dropout(drop_r)
        self.drop_r = drop_r


    def forward(self, x, y, x_mask, actions=None, log=False):
        """
        x = batch * len * hdim
        y = batch * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        y_flat = y.unsqueeze(1).expand(y.size(0), x.size(1), y.size(1)).contiguous().view(-1, y.size(-1))
        x_t = self.w1(x_flat)
        y_t = self.w2(y_flat)
        inpt = F.tanh(x_t+y_t)
        if self.drop_r>0:
            inpt = self.dropout(inpt)
        scores = self.v(inpt).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        if not log:
            return F.softmax(scores, dim=-1)
        else:
            return F.log_softmax(scores, dim=-1)


class GramMatrix(nn.Module):

    def forward(self, features):
        a, b = features.size()
        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b)


class GramMatrix_u(nn.Module):

    def forward(self, features):
        # features [batch x hid]
        a, b = features.size()
        G = torch.mul(features, features).sum(1)  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(b)


class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss(reduce=False)

    def forward(self, input):
        loss = self.criterion(input * self.weight, self.target)
        return loss


class StyleLoss(nn.Module):

    def __init__(self, target, weight, u=False):
        # u True: target  [batch x hid]
        # u False: target [hid1 x hid2]
        super(StyleLoss, self).__init__()
        self.target = target * weight
        self.weight = weight
        if u:
            self.gram = GramMatrix_u()
        else:
            self.gram = GramMatrix()
        self.criterion = nn.MSELoss(reduce=False)

    def forward(self, input):
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        loss = self.criterion(self.G, self.target)
        return loss


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1, keepdim=True).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


def make_action(probs):
    # sample from multinomial discrete distribution
    m = Categorical(probs.contiguous())
    actions = m.sample()
    logp = m.log_prob(actions)
    return actions, logp


def one_hot(actions, n_actions):
    #assert len(actions.size()) == 1
    a_onehot = torch.FloatTensor(actions.size(0), n_actions).cuda()
    a_onehot.zero_()
    try:
        a_onehot.scatter_(1, actions.data.unsqueeze(1), 1)
    except:
        a_onehot.scatter_(1, actions.unsqueeze(1), 1)
    return Variable(a_onehot)


def cat_entropy(logits,eps=1e-8):
    max_logits, _ = torch.max(logits, dim=1)
    a0 = logits - max_logits.unsqueeze(1)
    ea0 = a0.exp()
    z0 = ea0.sum(1).unsqueeze(1)
    p0 = ea0 / z0
    return (p0 * ((z0+eps).log() - a0)).sum(1)


def make_samples_concrete(logits, s, log_temp, eps=1e-8):
    u1 = Variable(torch.from_numpy(np.random.random(s)).float().cuda())
    u2 = Variable(torch.from_numpy(np.random.random(s)).float().cuda())
    temp = log_temp.exp()
    logprobs = F.log_softmax(logits, dim=-1)
    # gumbel random variable
    g = -(-(u1 + eps).log() + eps).log()
    # gumbel trick to sample max from categorical distribution
    scores = logprobs + g
    _, hard_samples = scores.max(1)
    hard_samples_oh = one_hot(hard_samples, scores.size(1))
    logprobs_z = scores

    g2 = -(-(u2 + eps).log() + eps).log()
    scores2 = logprobs + g2
    B = (scores2 * hard_samples_oh).sum(1).unsqueeze(1) - logprobs
    y = -1. * (u2).log() + (-1. * B).exp()
    g3 = -1. * (y).log()
    scores3 = g3 + logprobs
    # slightly biased…
    logprobs_zt = hard_samples_oh * scores2 + ((-1. * hard_samples_oh) + 1.) * scores3
    return hard_samples, F.softmax(logprobs_z / temp, dim=-1), F.softmax(logprobs_zt / temp, dim=-1)


def score_sc(pred_s, pred_m, truth):
    f1_s, f1_m = [], []
    if pred_s:
        assert len(pred_s) == len(truth)
        for ps, pm, t in zip(pred_s, pred_m, truth):
            f1_s += [_f1_score(ps, t)]
            f1_m += [_f1_score(pm, t)]
    else:
        for pm, t in zip(pred_m, truth):
            f1_m += [_f1_score(pm, t)]
    return np.array(f1_s), np.array(f1_m)


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))