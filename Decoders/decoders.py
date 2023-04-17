import torch
from torch import nn

''' DESCRIPTION 
In this file, we include 3 types of decoder:
    1) Last Cell Decoder (NOT USED)
    2) Standard sequence decoder + Projection;
    3) Standard sequence decoder + Projection + Bi-level
Copyright @ Lun Li
'''

'''
Last Cell Decoder
'''
class DecoderLastCell(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(DecoderLastCell, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # linear functional
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, hidden_seq):
        # get the last element of hidden seq: [batch_size, hidden_dim]
        out = hidden_seq[:, -1, :]
        # linear transformation
        out = self.fc(out)
        
        return out
    
'''
Standard Sequence Decoder + Projection
'''
class DecoderSeq(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, meta_data):
        super(DecoderSeq, self).__init__()
        self.meta_data = meta_data
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        # linear functional (Projection Matrix)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, hidden_seq, seq_len):
        # hidden_seq : [batch_size, seq_size, hidden_dim]
        # seq_len : [batch_size]
        concatenated = torch.cat([hidden_seq[i, : seq_len[i] - 1, :] for i in range(len(seq_len))], axis=0)
        v = self.fc(concatenated) # [selected_len, hidden_size] * [hidden_size, input_size] => selected_len * input_size
        return torch.matmul(v, self.meta_data.T)

'''
Standard Sequence Decoder + Projection + Bilevel
'''
class DecoderSeqBiLevel(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(DecoderSeq, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # linear functional (Projection Matrix)
        self.fc_finer = nn.Linear(self.hidden_size, self.output_size)
        self.fc_coarse = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, hidden_seq, seq_len):
        # hidden_seq : [batch_size, seq_size, hidden_dim]
        # seq_len : [batch_size]
        concatenated = torch.cat([hidden_seq[i, : seq_len[i] - 1, :] for i in range(len(seq_len))], axis=0)
        out = self.fc_finer(concatenated) # [selected_len, hidden_size] * [hidden_size, input_size] => selected_len * input_size
        # TODO: add diversity driven prob
        # v = self.fc_coarse(concatenated) # [selected_len, hidden_size] * [hidden_size, input_size] => selected_len * input_size
        return out

'''
Standard Sequence Decoder + Projection + Bi-Level
'''
class DecoderSeqAdvanced(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, meta_data, prob_hid):
        super(DecoderSeqAdvanced, self).__init__()
        self.meta_data = meta_data
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        # linear functional (Projection Matrix)
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        # ANN to model the probability of migrating away from current group
        # migrating prob(div, num_yr_worked, proj_emb)
        # Tried diff ways, it actually turns out (div, num_yr_worked) are sufficient
        # adding proj_emb obscure the effect
        # but no matter what, the contribution of modeling diversity prob is not much
        # To explain the residual, we shall find other source of info (e.g., meta path driven title recommendation etc)
        self.layers = nn.Sequential(
            nn.Linear(2 + self.input_size, prob_hid),
            nn.ReLU(),
            nn.Linear(prob_hid, 1))

    def forward(self, hidden_seq, seq_len, gp_sim_mask, div, yrs):

        # hidden_seq : [batch_size, seq_size, hidden_dim]
        # seq_len : [batch_size]
        # gp_sim_mask : [batch_size, seq_size, 4]
        
        ###
        # Due to the monotonicity of SOFTMAX / SIGMOID, all that matters are:
        # 1) the angle between current title and all titles;
        # 2) the sim-tile-transfer implied weights
        # 3) the diversity indicator imiplied weigts
        # We return 1) and 2) * 3) in the log space, i.e., log uncond prob, log cond prob, 
        # Taking exp(sum of log) recovers the total probability, this is done in the loss calculation
        ###

        ### This renders the raw inner product of current input and metaData (metaData is the embedding of all titles)
        # projection (emply -> title)
        proj_emb = self.fc(hidden_seq)
        # concatenation: [selected_len, hidden_size] * [hidden_size, input_size] => selected_len * input_size 
        concatenated = torch.cat([proj_emb[i, : seq_len[i] - 1, :] for i in range(len(seq_len))], axis=0) 
        # calculate the angle
        raw = torch.matmul(concatenated, self.meta_data.T)

        # 1) Sim-Title-Transfer
        # The similar group transfer is encoded in gp_sim_mask, which is pre-pocessed. To explain:
        # Gp_Sim_Mask is a binary vector over all categories, where 1 denotes current gp and sim group, 0 denotes the rest
        # For instance, mask1 identifies current gp + sim gp
        # Mask1:    | 1  1  1  1  |  1  1  1 |  1  1  1   1  |  0   0   0   0   0 |  0   0    0   0 |
        # Granular: | t1 t2 t3 t4 | t5 t6 t7 | t8 t9 t10 t11 | t11 t12 t13 t14 t15| t16 t17  t18 t19|
        # Coarse:   |   Cur Gp    |  Sim Gp1 |    Sim Gp2    |      Other GP 1    |    Other Gp2    |
        # Obviously, 1 - mask1 identifies the other gp1, i.e., 
        # Mask1^C:  | 0  0  0  0  |  0  0  0 |  0  0  0   0  |  1   1   1   1   1 |  1   1    1   1 |
        # Granular: | t1 t2 t3 t4 | t5 t6 t7 | t8 t9 t10 t11 | t11 t12 t13 t14 t15| t16 t17  t18 t19|
        # 
        # Notice: 
        # The structure also works if, instead of 0/1, we wanted to have weights, i.e.,
        # Mask1:     | 0.5  0.5  0.5  0.5  |  0.3  0.3  0.3 |  0.2  0.2  0.2   0.2  |  0   0   0   0   0  |  0   0    0   0 |
        # Mask1:     | 0    0    0    0    |  0    0    0   |  0    0    0     0    |  1   1   1   1   1  |  1   1    1   1 |
        
        # 2) Diversity Indicator
        # The migration probability is modelled through ANN + Sigmoid (div, num_yrs worked, cur employ proj emb)
        # which renders a prob over granular categories as below:
        # Mask2:    |      ...        1-p         ...        |        ...       p        ...        |
        # Granular: | t1 t2 t3 t4 | t5 t6 t7 | t8 t9 t10 t11 | t11 t12 t13 t14 t15| t16 t17  t18 t19|
        max_len = proj_emb.shape[1]
        factors_for_prob = torch.cat((div[:, :max_len, :], yrs[:, :max_len, :], proj_emb), axis=2) # [batch_size, max_seq_size, num_factors]
        ps = torch.sigmoid(self.layers(factors_for_prob))

        # This is a combined mask to account for both Sim Transfer and Diversity Indicator
        gp_sim_weighted = (1. - gp_sim_mask[:, :max_len, :]) * ps + gp_sim_mask[:, :max_len, :] * (1. - ps)
        # Notice : weighted version does not perform better than binary (COMMENTED OUT BELOW).
        # gp_sim_weighted = (gp_sim_mask == 0.).float() * ps + gp_sim_mask * (1. - ps)
        # This stretch all batches to a single vector
        concatenated_weights = torch.cat([gp_sim_weighted[i, : seq_len[i] - 1, :] for i in range(len(seq_len))], axis=0)
        
        # return angle and log weights in the exponent space
        return (raw, torch.log(concatenated_weights))