import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable    


''' DESCRIPTION 
PyTorch RNN structure is implemented in C for efficiency
We re-implement it in pytorch and extended to time-aware version.
This file includes:
    1) Standard GRU Cell Analytics;
    2) Standard LSTM Cell Analytics;
    3) Time-Aware GRU Cell Analytics;
    4) Time-Aware LSTM Cell Analytics
Copyright @ Lun Li
'''

'''
STANDARD GRU CELL
'''
class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # h_candidate = tanh ( W_c_h * h + W_c_x * x + b_c )
        # gamma_u     = sigma( W_u_h * h + W_u_x * x + b_u )
        # gamma_r     = sigma( W_r_h * h + W_r_x * x + b_r )
        # stacked memory linear transofrm, i.e., [W_c_h * h, W_u_h * h, W_r_h * h]
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # stacked input linear transform, i.e.,  [W_c_x * x + b_c, W_u_x * x + b_u, W_r_x * x + b_r]
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=True)
        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        # random initialize
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden=None):
        # Input Dimensions:
        # x:        (batch_size, input_size)
        # hidden:   (batch_size, hidden_size)

        if hidden is None:
            # initialize hidden to 0 if not existing
            hidden = Variable(x.new_zeros(x.size(0), self.hidden_size))

        # linear transformation
        gate_x = self.x2h(x) # (batch_size, hidden_size * 3)
        gate_h = self.h2h(hidden) # (batch_size, hidden_size)
        
        # get back each component
        # reset, input, candidate, each of them has dimension (batch_size, hidden_size)
        x_reset, x_upd, x_new = gate_x.chunk(3, 1) # [W_r_x * x + b_r, W_u_x * x + b_u, W_r_x * x + b_r]
        h_reset, h_upd, h_new = gate_h.chunk(3, 1) # [W_r_h * h, W_u_h * h, W_r_h * h]
        
        # nonlinear transformation
        reset_gate = F.sigmoid(x_reset + h_reset) 
        input_gate = F.sigmoid(x_upd + h_upd)
        candidate = F.tanh(x_new + (reset_gate * h_new))
        
        # output : (1.- input_gate) * candidate + input_gate * hidden
        # written as below for computational efficiency
        o = candidate + input_gate * (hidden - candidate)

        # Output Dimension:
        # o:        (batch_size, hidden_size)
        return o

'''
STANDARD LSTM CELL
'''
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # linear functionals
        # i_t =       sigma (W_i_h * h + W_i_x * x + b_i)
        # f_t =       sigma (W_f_h * h + W_f_x * x + b_f)
        # o_t =       sigma (W_o_h * h + W_o_x * x + b_o)
        # candidate = tanh  (W_c_h * h + W_c_x * x + b_c)
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.xh = nn.Linear(input_size, hidden_size * 4, bias=False)
        # reset
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden=None):
        # Input Dimensions:
        # x:        (batch_size, input_size)
        # hidden:   (batch_size, hidden_size)

        # unfold hidden (hidden + memory)
        if hidden is None:
            hidden = Variable(x.new_zeros(x.size(0), self.hidden_size))
            hidden = (hidden, hidden) 
        else:
            hidden, memory = hidden

        # execute all linear transformation
        gates = self.xh(x) + self.hh(hidden)

        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        # execute nonlinear transformation
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        o_t = torch.sigmoid(output_gate)
        candidate_t = torch.tanh(cell_gate)

        # memory
        memory_ = memory * f_t + i_t * candidate_t

        # hidden
        hidden_ = o_t * torch.tanh(memory_)

        # Output Dimension:
        # hidden_:        (batch_size, hidden_size)
        # memory_:        (batch_size, hidden_size)
        return (hidden_, memory_)

'''
TIME-AWARE GRU
'''
class TGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(TGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # h_candidate = tanh ( W_c_h * h + W_c_x * x + b_c )
        # gamma_u     = sigma( W_u_h * h + W_u_x * x + b_u )
        # gamma_r     = sigma( W_r_h * h + W_r_x * x + b_r )
        # stacked memory linear transofrm, i.e., [W_c_h * h, W_u_h * h, W_r_h * h]
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # stacked input linear transform, i.e.,  [W_c_x * x, W_u_x * x, W_r_x * x]
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=False)
        # discount ratio r, features in exp{-r\Delta_n}
        # To add degree of freedom, we allow r * \Delta_n to be pointwise
        self.r = torch.nn.Parameter(torch.randn(1, hidden_size))
        # inflating ratio theta, features in exp{\theta\Delta_n}
        # To add degree of freedom, we allow \theta * \Delta_n to be pointwise
        self.theta = torch.nn.Parameter(torch.randn(1, hidden_size))
        # Independent Biase (WE DON"T FUSE IT INTO LINEAR FUNCTIONAL INTENTIONALLY)
        self.bias_r = torch.nn.Parameter(torch.randn(1, hidden_size))
        self.bias_u = torch.nn.Parameter(torch.randn(1, hidden_size))
        self.bias_c = torch.nn.Parameter(torch.randn(1, hidden_size))
        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        # random initialize
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden=None):
        
        # Input Dimensions:
        # x:        (batch_size, input_size)
        # hidden:   (batch_size, hidden_size)

        x_, delta_t = x[0], x[1]
        
        if hidden is None:
            # initialize hidden to 0 if not existing
            hidden = Variable(x_.new_zeros(x_.size(0), self.hidden_size))

        # linear transformation
        gate_x = self.x2h(x_) # (batch_size, hidden_size * 3)
        gate_h = self.h2h(hidden) # (batch_size, hidden_size * 3)
        
        # get back each component
        # reset, input, candidate, each of them has dimension (batch_size, hidden_size)
        x_reset, x_upd, x_new = gate_x.chunk(3, 1) # [W_r_x * x, W_u_x * x, W_r_x * x]
        h_reset, h_upd, h_new = gate_h.chunk(3, 1) # [W_r_h * h, W_u_h * h, W_r_h * h]
        
        # duration based weight
        integrand_i = torch.pow(self.theta, 2.) * delta_t

        # nonlinear transformation
        reset_gate = F.sigmoid(x_reset + h_reset + self.bias_r) 
        input_gate = F.sigmoid(x_upd + h_upd + self.bias_u)
        candidate = F.tanh(x_new * torch.exp(integrand_i)  + (reset_gate * h_new) + self.bias_c)
        
        # output : (1.- input_gate) * candidate + input_gate * hidden * df
        integrand_r = torch.pow(self.r, 2.) * delta_t
        o = (1.- input_gate) * candidate + input_gate * hidden * torch.exp(-integrand_r)

        # Output Dimension:
        # o:        (batch_size, hidden_size)
        return o

'''
TIME-AWARE LSTM
'''
class TLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # linear functionals
        # i_t =       sigma (W_i_h * h + W_i_x * x + b_i)
        # f_t =       sigma (W_f_h * h + W_f_x * x + b_f)
        # o_t =       sigma (W_o_h * h + W_o_x * x + b_o)
        # candidate = tanh  (W_c_h * h + W_c_x * x + b_c)
        # stacked memory linear transofrm, i.e., [W_i_h * h, W_f_h * h, W_o_h * h, W_c_h * h]
        self.hh = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        # stacked memory linear transofrm, i.e., [W_i_x * x, W_f_x * x, W_o_h * x, W_c_h * x]
        self.xh = nn.Linear(input_size, hidden_size * 4, bias=False)
        # discount ratio r, features in exp {-r\Delta_n}
        self.r = torch.nn.Parameter(torch.randn(1, hidden_size))
        # inflating ratio theta, features in exp{\theta\Delta_n}
        self.theta = torch.nn.Parameter(torch.randn(1, hidden_size))
        # Independent Biase (WE DON'T FUSE IT INTO LINEAR FUNCTIONAL INTENTIONALLY)
        self.bias_i = torch.nn.Parameter(torch.randn(1, hidden_size))
        self.bias_f = torch.nn.Parameter(torch.randn(1, hidden_size))
        self.bias_o = torch.nn.Parameter(torch.randn(1, hidden_size))
        self.bias_c = torch.nn.Parameter(torch.randn(1, hidden_size))
        # reset
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden=None):

        # Input Dimensions:
        # x:        (batch_size, input_size)
        # hidden:   (batch_size, hidden_size)

        x_, delta_t = x[0], x[1]

        # unfold hidden (hidden + memory)
        if hidden is None:
            hidden = Variable(x_.new_zeros(x.size(0), self.hidden_size))
            hidden = (hidden, hidden) 
        else:
            hidden, memory = hidden

        # execute all linear transformation
        gate_x = self.xh(x_) # (batch_size, hidden_size * 4)
        gate_h = self.hh(hidden) # (batch_size, hidden_size * 4)

        # Get gates (i_t, f_t, o_t, c_t)
        x_input_gate, x_forget_gate, x_output_gate, x_cell_gate = gate_x.chunk(4, 1)
        h_input_gate, h_forget_gate, h_output_gate, h_cell_gate = gate_h.chunk(4, 1)

        # duration based weight
        integrand_i = torch.pow(self.theta, 2.) * delta_t

        # execute nonlinear transformation
        i_t = F.sigmoid(x_input_gate + h_input_gate + self.bias_i) # i_t = torch.sigmoid(input_gate)
        f_t = F.sigmoid(x_forget_gate + h_forget_gate + self.bias_f) # f_t = torch.sigmoid(forget_gate)
        o_t = F.sigmoid(x_output_gate + h_output_gate + self.bias_o) # o_t = torch.sigmoid(output_gate)
        candidate_t = F.tanh(x_cell_gate * torch.exp(integrand_i) + h_cell_gate + self.bias_c) # candidate_t = torch.tanh(cell_gate)

        # memory
        integrand_r = torch.pow(self.r, 2.) * delta_t
        memory_ = memory * f_t * torch.exp(-integrand_r) + i_t * candidate_t

        # hidden
        hidden_ = o_t * torch.tanh(memory_)

        # Output Dimension:
        # hidden_:        (batch_size, hidden_size)
        # memory_:        (batch_size, hidden_size)
        return (hidden_, memory_)