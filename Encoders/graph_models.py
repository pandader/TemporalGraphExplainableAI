import torch
from torch_geometric.nn import GCNConv


''' DESCRIPTION 
1) A customized PyG GCN for title embedding
2) A simple HAN based on metapath
Copyright @ Lun Li
'''

### simple 2-layer gcn
class GCN(torch.nn.Module):
    
    def __init__(self, num_features, hidden_dim, out_dim, graph_lst, use_raw_emb=True):
        super().__init__()
        self.use_raw_emb = use_raw_emb
        self.graph_lst = graph_lst
        self.gcn_l1 = GCNConv(num_features, hidden_dim, cached=True)
        self.gcn_l2 = GCNConv(hidden_dim, out_dim, cached=True)
        
    def forward(self):
        if self.use_raw_emb:
            return x
        res_g = []
        for g in self.graph_lst:
            x, edge_index, edge_weight = g.x, g.edge_index, g.edge_attr            
            h = self.gcn_l1(x, edge_index, edge_weight).relu()
            o = self.gcn_l2(h, edge_index, edge_weight)
            res_g.append(o)
        return res_g

    def filter(self, title_emb, emp_ids, end_yrs, dynamicTitleEmb):
        ### TODO: filtering
        return title_emb



### simple HAN (parameter-free)
### TODO: adding parameters to customize embedding aggregation
class HANMeta(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, df_metapath, title_emb_mat, emp_ids, end_yrs, batch_label, inputs):
        # inputs = outputs from rnn
        title_emb_dim = title_emb_mat.shape[1]
        emb_updated = []
        for i in range(batch_label.shape[0]):
            focal_id = 'E' + str(int(emp_ids[i]))
            tmp_ref = df_metapath[(df_metapath.Focal == focal_id)]
            graph_concat = torch.zeros(inputs.shape[1], title_emb_dim)
            for pos in range(inputs[i].shape[0]):
                yr = end_yrs[i][pos]
                if yr == 0: continue
                focal_cur_yr = tmp_ref[tmp_ref.Year == yr]
                if len(focal_cur_yr) == 0: continue
                ref_embs = inputs[focal_cur_yr.BatchPos.values, focal_cur_yr.Job_Index.values]
                focal_emb = inputs[i][pos]
                raw_s = torch.inner(focal_emb.unsqueeze(0), ref_embs)
                sim = (raw_s.exp() / raw_s.exp().sum())
                ref_t_embs = title_emb_dim[[int(each.split('T')[1]) for each in focal_cur_yr.TargetTitle]]
                graph_t_emb = torch.matmul(sim, ref_t_embs)
                graph_concat[pos] = graph_t_emb.squeeze()
            emb_updated.append(torch.concat([inputs[i], graph_concat], axis=1))
        return torch.cat(emb_updated, axis=0)
        


