import os
import json
import numpy as np
import pandas as pd

def loadTitleEmbedding(importPath, importFile, title_map_dict):
    ### load title embedding
    np_title_emb = np.load(os.path.join(importPath, importFile), allow_pickle=True)
    title_2_emb = {
        title_map_dict[t] : e for t, e, in np_title_emb if t in title_map_dict}
    return np.array([title_2_emb['T' + str(i)] for i in range(1, len(title_2_emb) + 1)])

def loadEntityMappings(import_path, import_file):
    ### import title/employee id dict
    with open(os.path.join(import_path, import_file), "r") as openfile:
        mapped_dict = json.load(openfile)
    return mapped_dict

def loadGoldenSource(importPath, importFile):
    ### import golden source - cleaned profiles
    df = pd.read_csv(os.path.join(importPath, importFile))
    df['Start_Year'] = df['Start_Date'].apply(lambda x: int(x.split('-')[0]))
    df['End_Year'] = df['End_Date'].apply(lambda x: int(x.split('-')[0]))
    df.drop_duplicates(subset=['ID', 'End_Year'], keep='last', inplace=True)
    df.drop_duplicates(subset=['ID', 'Start_Year'], keep='last', inplace=True)
    org_employee_id_2_new_id = {id : "E" + str(idx + 1) for idx, id in enumerate(df.ID.unique())}
    df['EmployeeID'] = df.ID.apply(lambda x: org_employee_id_2_new_id[x])
    org_title_2_new_id = {id : "T" + str(idx + 1) for idx, id in enumerate(df.Title.unique())}
    df['TitleID'] = df.Title.apply(lambda x: org_title_2_new_id[x])
    df['Job_Index'] = df.groupby('EmployeeID').cumcount()
    return df
