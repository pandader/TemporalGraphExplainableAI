import torch
from torch import nn
from enum import Enum
from torch.autograd import Variable
### in-house
from Encoders.rnn_cell import GRUCell, TGRUCell, LSTMCell, TLSTMCell

''' DESCRIPTION 
In this file, we assign the analytics in rnn_cell.py to corresponding encoders.
Here is what encoder does:

    INPUT                     RNN_CELL                          OUTPUT        
------------------------------------------------------------------------------
sequence data ---> [ cell analytics in rnn_cell.py ] ---> sequential embedding
------------------------------------------------------------------------------

Notice: the pyTorch native LSTM/RNN API implemented dynamic rolling in C. Here,
        we replicate it with masking.

Copyright @ Lun Li
'''

'''
ENUMS: for RNN type being either Standard or Time-Aware
'''
class RNN_CELL_TYPE(Enum):
    PLAIN = 1 # standard
    TIME_AWARED = 2 # time-aware

'''
ENCODER: GRU
'''
class GRU(nn.Module):

    CELL_TYPE = {
        RNN_CELL_TYPE.PLAIN : GRUCell,
        RNN_CELL_TYPE.TIME_AWARED : TGRUCell
    }

    def __init__(self, input_size, hidden_size, num_layers, cell_type=RNN_CELL_TYPE.PLAIN):
        super(GRU, self).__init__()

        ### TODO: EXTEND IT TO MULTIPLE LAYER (the pipeline is in place)
        if num_layers > 1:
            raise Exception('Only supports single layer GRU')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        # allow multiple layers
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(GRU.CELL_TYPE[self.cell_type](self.input_size, self.hidden_size))
        for _ in range(1, self.num_layers):
            self.rnn_cell_list.append(GRU.CELL_TYPE[self.cell_type](self.hidden_size, self.hidden_size))

    def forward(self, input, hidden=None):

        # Input Dimension: (batch_size, sequence_size, input_size)

        input_, seq_len = input[0], input[1]
        max_len = seq_len.max().int()

        # in case there is no initial inputs
        if hidden is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input_.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input_.size(0), self.hidden_size)) 
        else:
             h0 = hidden # [num_layers, batch_size, hidden_size], e.g., [1, 3, 128]

        # collector
        outs = torch.zeros(max_len, input_.shape[0], self.hidden_size)
        hx = h0[0, :, :]
        for t in range(max_len):
            if self.cell_type == RNN_CELL_TYPE.PLAIN:
                hidden_l = self.rnn_cell_list[0](input_[:, t, :], hx)
            else:
                hidden_l = self.rnn_cell_list[0]((input_[:, t, :], input[2][:, t, :]), hx)
            # masking logic
            mask = (t < seq_len).float().unsqueeze(1).expand_as(hidden_l)
            h_next = hidden_l*mask + hx*(1 - mask)
            outs[t] = h_next
            hx = h_next

        return outs.permute(1, 0, 2)

'''
ENCODER: LSTM
'''
class LSTM(nn.Module):

    CELL_TYPE = {
        RNN_CELL_TYPE.PLAIN : LSTMCell,
        RNN_CELL_TYPE.TIME_AWARED : TLSTMCell
    }

    def __init__(self, input_size, hidden_size, num_layers, cell_type=RNN_CELL_TYPE.PLAIN):
        super(LSTM, self).__init__()

        ### TODO: EXTEND IT TO MULTIPLE LAYER (the pipeline is in place)
        if num_layers > 1:
            raise Exception('Only supports single layer LSTM')
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        # allow multiple layers
        self.rnn_cell_list = nn.ModuleList()
        self.rnn_cell_list.append(LSTM.CELL_TYPE[self.cell_type](self.input_size, self.hidden_size))
        for _ in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTM.CELL_TYPE[self.cell_type](self.hidden_size, self.hidden_size))

    def forward(self, input, hidden=None):

        # Input Dimension: (batch_size, seqence_size , input_size)
        input_ = input[0]
        seq_len = input[1]
        max_len = seq_len.max().int()

        # in case there is no initial inputs
        if hidden is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input_.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input_.size(0), self.hidden_size))
        else:
             h0 = hidden

        # collectors
        outs = torch.zeros(max_len, input_.shape[0], self.hidden_size)
        # unrolling
        hx = (h0[0, :, :], h0[0, :, :]) # h and c
        for t in range(max_len):
            hidden_l = self.rnn_cell_list[0](input_[:, t, :],  hx) if self.cell_type == RNN_CELL_TYPE.PLAIN \
                else self.rnn_cell_list[0]((input_[:, t, :], input[2][:, t, :]), hx)
            # masking technique
            mask = (t < seq_len).float().unsqueeze(1).expand_as(hidden_l[0])
            h_next = hidden_l[0]*mask + hx[0]*(1 - mask)
            c_next = hidden_l[1]*mask + hx[1]*(1 - mask)
            hx_next = (h_next, c_next)
            outs[t] = h_next
            hx = hx_next
            
        return outs.permute(1, 0, 2)