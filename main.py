import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.autograd import Variable

### in-house
from utilities.gen_utilities import (init_torch, CustomDataSet, getMetaPathInducedGraph, loadPyGeoData)
from Encoders.rnn_kernels import (RNN_CELL_TYPE)
from Encoders.master_model import (RNN_TYPES, MasterModel)

#################################################################################################

### initialize pytorch
init_torch()

'''
USER CONFIGS
'''
DEBUG_MODE = False # KEPPS DEBUG = TRUE, to design model, it uses a very small data set
IMPORT_PATH = '\TemporalGraphData'
RNN_TYPE = RNN_TYPES.LSTM # Available Choices: RNN_TYPES.GRU, RNN_TYPES.LSTM
CELL_TYPE = RNN_CELL_TYPE.PLAIN # Available Choices: RNN_CELL_TYPE.PLAIN, RNN_CELL_TYPE.TIME_AWARED
BATCH_SIZE = 1000 if DEBUG_MODE else 1000
HIDDEN_DIM = 128 # RNN cell
N_ITER = 80 if DEBUG_MODE else 600
USE_BI_LEVEL = False
TOPK = 5 # accuracy measure, ie., Acc@TopK


if __name__ == "__main__":

    '''
    STEP 1: LOAD DATA
    '''
    train_dataset, test_dataset = None, None
    tmp_path = os.path.join(IMPORT_PATH, 'TrainTestData')
    if DEBUG_MODE:
        train_dataset = CustomDataSet(tmp_path, 'DEBUG_X.pt', 'DEBUG_Y.pt')
        test_dataset = CustomDataSet(tmp_path, 'DEBUG_TEST_X.pt', 'DEBUG_TEST_Y.pt')
    else:
        train_dataset = CustomDataSet(tmp_path, 'TRAIN_X.pt', 'TRAIN_Y.pt')
        test_dataset = CustomDataSet(tmp_path, 'TEST_X.pt', 'TEST_Y.pt')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    num_train_samples = len(train_dataset)
    # load org title embedding (not necessary)
    v = np.load(os.path.join(IMPORT_PATH, 'title_emb_matrix.npy'), allow_pickle=True)
    title_emb_tensor = Variable(torch.FloatTensor(v))
    # load dynamic title evolving seq
    title_dgs = loadPyGeoData(IMPORT_PATH, "TitleGraphs", 'pyg_title_{}.pt', list(range(2008, 2023)))
    # auxiliary data for meta path (ETET)
    df_etet = pd.read_csv(os.path.join(IMPORT_PATH, 'ETETGraphs\AggregatedETET.csv'))

    '''
    STEP 2: INSTANTIATE MODEL
    '''
    num_titles = title_emb_tensor.shape[0]
    input_title_dim = title_emb_tensor.shape[1] # Input Title Dim
    num_epochs = N_ITER/(num_train_samples/BATCH_SIZE)
    num_epochs = int(num_epochs)
    model = MasterModel(
        input_title_dim, input_title_dim, HIDDEN_DIM, num_titles, HIDDEN_DIM, title_dgs, RNN_TYPE, cell_type=CELL_TYPE)

    '''
    STEP 3: INSTANTIATE LOSS FUNCTION & OPTIMIZER
    '''
    logSoftMax = None
    learning_rate = 0.1
    if USE_BI_LEVEL:
        logSoftMax=nn.LogSoftmax(dim=1)
        criterion = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

    '''
    STEP 4: TRAIN / TEST THE MODEL
    '''
    loss_list = []
    iter = 0
    for epoch in range(num_epochs):
        for i, (emp_id, num_exp, end_yr, job_idx, dur, div, t_emb, label) in enumerate(train_loader):
            emp_id = emp_id.int().numpy()
            if torch.cuda.is_available():
                t_emb = Variable(t_emb.cuda())
                num_exp_sq = Variable(num_exp.squeeze(-1).cuda())
                end_yr_sq = end_yr.squeeze().int().numpy()
                dur = Variable(dur.cuda())
                div = Variable(div.cuda())
                out_label_seq = Variable(label.long().cuda())
            else:
                t_emb = Variable(t_emb)
                num_exp_sq = Variable(num_exp.squeeze(-1).cuda())
                end_yr_sq = end_yr.squeeze().int().numpy()
                dur = Variable(dur)
                div = Variable(div)
                out_label_seq = Variable(label.long())
            

            # get metapath recommender for the given batch
            df_helper = getMetaPathInducedGraph(emp_id, df_etet)

            # reset dModel/dParams
            optimizer.zero_grad()

            # Calculate Loss: softmax --> cross entropy loss
            outputs = model((t_emb, emp_id, end_yr_sq, num_exp_sq, label), df_helper)
            # benchmark
            seq_len = num_exp.squeeze(-1).int()
            output_title_seq_ = torch.cat([out_label_seq[i, 1 : seq_len[i]] for i in range(len(seq_len))], axis=0)
            # calc loss
            loss = criterion(outputs, output_title_seq_)
            if torch.cuda.is_available():
                loss.cuda()

            # update params
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            iter += 1

            if iter % 2 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                for i, (emp_id, num_exp, end_yr, job_idx, dur, div, t_emb, label) in enumerate(train_loader):
                    emp_id = emp_id.int().numpy()
                    if torch.cuda.is_available():
                        t_emb = Variable(t_emb.cuda())
                        num_exp_sq = Variable(num_exp.squeeze(-1).cuda())
                        end_yr_sq = end_yr.squeeze().int().numpy()
                        dur = Variable(dur.cuda())
                        div = Variable(div.cuda())
                        out_label_seq = Variable(label.long().cuda())
                    else:
                        t_emb = Variable(t_emb)
                        num_exp_sq = Variable(num_exp.squeeze(-1).cuda())
                        end_yr_sq = end_yr.squeeze().int().numpy()
                        dur = Variable(dur)
                        div = Variable(div)
                        out_label_seq = Variable(label.long())
                

                # get metapath recommender for the given batch
                df_helper = getMetaPathInducedGraph(emp_id, df_etet)

                # Calculate Loss: softmax --> cross entropy loss
                outputs = model((t_emb, emp_id, end_yr_sq, num_exp_sq, label), df_helper)
                # benchmark
                seq_len = num_exp.squeeze(-1).int()
                output_title_seq_ = torch.cat([out_label_seq[i, 1 : seq_len[i]] for i in range(len(seq_len))], axis=0)
                
                top_res = outputs.topk(TOPK).indices
                total += outputs.size()[0]
                correct += np.sum([(output_title_seq_[i] in top_res[i]) for i in range(outputs.size()[0])])
                accuracy = 100. * correct / total
                
                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
                