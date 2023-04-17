import os
import torch
import numpy as np
from torch.utils.data import Dataset

### initialize pyTorch environment
def init_torch(seed=25):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### load pyG data
def loadPyGeoData(folder, path, file_name, post_fix = []):
    path_ = os.path.join(folder, path)
    if len(post_fix) == 0:
        return torch.load(os.path.join(path_, file_name))
    data_lst = []
    for each in post_fix:
        data_lst.append(torch.load(os.path.join(path_, file_name.format(each))))
    return data_lst

### data loader
class CustomDataSet(Dataset):

    def __init__(self, root_path, x_file, y_file, title_emb_dim=384, use_cuda=False):
        self.dataset = torch.load(os.path.join(root_path, x_file),)
        self.labels = torch.load(os.path.join(root_path, y_file))
        self.title_emb_dim = title_emb_dim
        if use_cuda:
            self.dataset = self.dataset.cuda()
            self.labels = self.labels.cuda()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx], self.labels[idx]
        emp_id = x[:, 0]
        num_exp = x[:, 1:2]
        end_yr = x[:, 2:3]
        job_idx = x[:, 3:4]
        dur = x[:, 4:5]
        div = x[:, 5:6]
        t_emb = x[:, 6:]
        return emp_id[0], num_exp[0], end_yr, job_idx, dur, div, t_emb, y[:, 0]
    
### get etet path recommender
# give a focal employee in the batch
# at every career transition time
# get the reference employees and his/her recommended titles
def getMetaPathInducedGraph(batch_emp_ids, df_etet):
    helper_dict = {'E' + str(int(id)) : idx for idx, id in enumerate(batch_emp_ids)}
    relevant = ['E' + str(id) for id in batch_emp_ids]
    df_helper = df_etet[(df_etet.Focal.isin(relevant))&(df_etet.Reference.isin(relevant))]
    df_helper['BatchPos'] = df_helper.Reference.apply(lambda x: helper_dict[x])
    return df_helper