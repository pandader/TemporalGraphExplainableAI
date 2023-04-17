import torch
import numpy as np
from torch import nn
from enum import Enum
# in house
from Encoders.graph_models import GCN, HANMeta
from Encoders.rnn_kernels import GRU, LSTM
from Decoders.decoders import DecoderSeqBiLevel

''' DESCRIPTION 
In this file, we set up a generic interface of RNN model that consists of encoder and decoder
The master model allow sequence model only as well as hybrid model that considers graph evolution info
Copyright @ Lun Li
'''

'''
RNN TYPE
'''
class RNN_TYPES(Enum):
    GRU = 1
    LSTM = 2

'''
Depends on the building block, we get different variation of encoder + decoder
    - type: gru, lstm
    - time-aware: true, false
    - enable graph: true, false
    - decoder type: ...
'''
class MasterModel(nn.Module):
    
    def __init__(
            self, input_size, title_gcn_out, seq_hiddn, output_size, title_gcn_hid = 128, title_g_data = [],
            rnn_type=RNN_TYPES.LSTM, num_layers=1, cell_type='PLAIN', use_raw_emb=True):
        super(MasterModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = seq_hiddn
        assert(num_layers == 1)
        self.num_layers = 1
        # initialize graphical encoder
        self.title_g_enco = GCN(self.input_size, title_gcn_hid, title_gcn_out, title_g_data, use_raw_emb)
        # initialize sequence encoder
        if rnn_type == RNN_TYPES.GRU:
            self.seq_enco = GRU(self.input_size, self.hidden_size, self.num_layers, cell_type)
        elif rnn_type == RNN_TYPES.LSTM:
            self.seq_enco = LSTM(self.input_size, self.hidden_size, self.num_layers, cell_type)
        # initialize meta path enhancer
        self.meta_enco = HANMeta()
        # initialize decoder
        self.decoder = DecoderSeqBiLevel(self.input_size + self.hidden_size, output_size)
        
        # to cuda
        if torch.cuda.is_available():
            self.encoder.cuda()
            # self.decoder.cuda()

    def forward(self, input, df_metapath):

        title_emb = input[0]
        emp_ids = input[1]
        end_yrs = input[2]
        num_exp_sq = input[3]
        batch_label = input[4]
        
        ### Title Generation (Dynamic)
        # get evolving title embedding
        z_raws = self.title_g_enco()
        ### TODO: re-assemble title emb seq for each employee in the batch
        ###       by filtering from z_raws (derived from GCN)
        title_emb = self.title_g_enco.filter(title_emb, emp_ids, end_yrs, z_raws)
        
        ### Typicality Embedding Generation
        outputs = self.seq_enco((title_emb, num_exp_sq))

        ### MetaPath Embedding Improvements
        hidden_seq = self.meta_enco(df_metapath, title_emb, emp_ids, end_yrs, batch_label, outputs)

        ### DECODER
        output = self.decoder(hidden_seq, num_exp_sq)

        return output