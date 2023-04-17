''' DESCRIPTION 
A time-indexed title graphs:
for any given yr, we create a undirected title flow among all titles (sum up bi-directional flows).
Copyright @ Lun Li
'''
import os
import torch
import warnings
import numpy as np
from torch_geometric.data import Data
from utils import (loadGoldenSource, loadEntityMappings, loadTitleEmbedding)
warnings.simplefilter('ignore')


### global var
START_YR = 2008
END_YR = 2023
EXCLUDE_YR = []
TITLE_EMB = 'title_only_emb.npy'
TITLE_MAPPING = 'title_id_dict.json'
IMPROT_FILE = 'validData_Reduced.csv'
IMPORT_PATH = '\TemporalGraphData'

def pyGCreate(df_agg, z_raw):
    ### create directed graph
    # raw title emb matrix
    x = torch.tensor(z_raw, dtype=torch.float)
    # edge index
    from_v = df_agg['From'].apply(lambda x: int(x.split('T')[1])-1).values
    to_v = df_agg['To'].apply(lambda x: int(x.split('T')[1])-1).values
    edge_index = torch.tensor(np.array([from_v, to_v]), dtype=torch.long)
    # edge attr
    edge_attr = torch.tensor(df_agg.Frequency.values.reshape(len(df_agg), 1), dtype=torch.float)
    # graph data
    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index.contiguous())


def processSingleYr(df, yr):
    # select yr = 2018
    df_yr = df[(df['Start_Year']==yr)|(df['End_Year']==yr)]
    df_yr_ = df_yr.join(df_yr.groupby('ID')['TitleID'].shift(-1),rsuffix='1')
    df_yr_.rename({'TitleID1': 'transferTo'}, axis=1, inplace=True)
    df_valid = df_yr_[df_yr_['transferTo'].notnull()]

    # aggregate flows
    df_valid['pair'] = df_valid.apply(
        lambda x: ','.join(np.sort([x['TitleID'], x['transferTo']])), axis = 1)
    df_agg = df_valid.groupby(["pair"]).size().reset_index(name="Frequency")
    df_agg[['From', 'To']] = df_agg['pair'].str.split(',', 1, expand=True)
    df_agg.drop(columns=['pair'], inplace=True)
    df_agg = df_agg[['From', 'To', 'Frequency']]
    df_agg = df_agg[df_agg['From'] != df_agg['To']]
    
    return df_agg

if __name__ == "__main__":
    export_path = os.path.join(IMPORT_PATH, 'TemporalGraphData\TitleGraphs')
    df_golden = loadGoldenSource(IMPORT_PATH, IMPROT_FILE)
    org_title_2_new_id = loadEntityMappings(IMPORT_PATH, TITLE_MAPPING)
    z_raw = loadTitleEmbedding(IMPORT_PATH, TITLE_EMB, org_title_2_new_id)

    df_golden['TitleID'] = df_golden.Title.apply(lambda x: org_title_2_new_id[x])
    for yr in range(START_YR, END_YR):
        if yr in EXCLUDE_YR:
            continue
        try:
            df_agg = processSingleYr(df_golden, yr)
            data = pyGCreate(df_agg, z_raw)
            # export
            torch.save(data, os.path.join(export_path, 'pyg_title_{yr}.pt'.format(yr=yr)))
            df_agg.to_csv(os.path.join(export_path, 'graph_title_{yr}.csv'.format(yr=yr)), index=False)
            print('File graph_title_{yr}.csv is exported'.format(yr=yr))
        except:
            # pd.DataFrame(columns=['From', 'To', 'Frequency']).to_csv(os.path.join(export_path, 'graph_title_{yr}.csv'.format(yr=yr)))
            print('Weird! File graph_title_{yr}.csv is exported'.format(yr=yr))
        