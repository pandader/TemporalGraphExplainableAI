''' DESCRIPTION 
A time-indexed HINs based on metapath ETET

For a given yr, and given lookback window:
1) we construct a ETET graph;
2) we apply two constraints: 1) non-anticipativity; 2) no obsolete titles;
3) we trim the graph so that each individual does not have repeated target titles;

Remark: the last step is done by comparing the duration of the shared job.
Copyright @ Lun Li
'''

import os
import warnings
import numpy as np
import pandas as pd
from utils import (loadGoldenSource)
warnings.simplefilter('ignore')

### global var
WINDOW_SIZE = 4
START_YR = 2008
END_YR = 2023
EXCLUDE_YR = []
IMPROT_FILE = 'validData_Reduced.csv'
IMPORT_PATH = '\TemporalGraphData'

def processSingleYr(yr, df, w_size):
    
    ### filter employees that make transition in year = yr
    df_cur = df[df.End_Year == yr].reset_index(drop=True)
    num = len(df_cur.EmployeeID.unique())
    print('There are {num} made career transition.'.format(num=num))
    print('0. Process year {yr}.'.format(yr=yr))

    ### get all profiles in the lookback window
    window_s, window_e = yr - w_size, yr
    cond1 = (df.End_Year >= window_s) & (df.End_Year <= window_e)
    cond2 = (df.Start_Year >= window_s) & (df.Start_Year < window_e)
    df_window = df[cond1|cond2].reset_index(drop=True)

    ### get ETE part through dataFrame level operation
    df_ete = df_cur[['EmployeeID', 'TitleID', 'Duration']].merge(
        df_window[['EmployeeID', 'TitleID', 'End_Year', 'Duration', 'Job_Index']], on='TitleID', how='left')
    df_ete = df_ete[df_ete.EmployeeID_x != df_ete.EmployeeID_y]
    print('1. Finished ETE Part.')

    ### complete ETE by T
    count = 0
    df_col = []
    gps = df_ete.groupby(['EmployeeID_x', 'TitleID'])
    for id, gp in gps:    
        df_ref_tmp = df_window[df_window.EmployeeID.isin(gp.EmployeeID_y)]
        tmp = df_ref_tmp.merge(
            gp[['EmployeeID_y', 'End_Year', 'Duration_x', 'Duration_y', 'Job_Index']], 
            left_on='EmployeeID', right_on='EmployeeID_y', how='left')
        tmp = tmp[(tmp.Start_Year >= tmp.End_Year_y)&(tmp.TitleID != id[1])][
            ['EmployeeID_y', 'TitleID', 'Start_Year', 'End_Year_x', \
             'Duration', 'EDU', 'Duration_x', 'Duration_y', 'Job_Index_x']]
        tmp['Focal'], tmp['SharedTitle'] = id[0], id[1]
        df_col.append(tmp)
        count += 1
        print(count)
    print('2. Found all ETET instances.')

    # clean up
    df_agg = pd.concat(df_col, axis=0)
    df_agg = df_agg[
        ['Focal', 'SharedTitle', 'Duration_x', 'Duration_y', 'EmployeeID_y', \
         'TitleID', 'Start_Year', 'End_Year_x', 'Duration', 'Job_Index_x']].rename(columns={\
         'EmployeeID_y' : 'Reference', 'TitleID' : 'TargetTitle', 'End_Year_x' : 'End_Year',
         'Job_Index_x' : 'Job_Index'})
    
    # reduce by sim (shared job worked similar yrs)
    df_agg.reset_index(drop=True, inplace=True)
    df_agg['sim'] = df_agg['Duration_x'] - df_agg['Duration_y']
    df_agg['sim'] = df_agg['sim'].apply(abs)
    df_agg_red = df_agg.loc[df_agg.groupby(['Focal', 'SharedTitle', 'TargetTitle']).sim.idxmin()]
    print('3. Graph trimed based on duration similairty.')
    
    return df_agg_red

if __name__ == "__main__":
    export_path = os.path.join(IMPORT_PATH, 'TemporalGraphData\ETETGraphs')
    df_golden = loadGoldenSource(IMPORT_PATH, IMPROT_FILE)
    for yr in range(START_YR, END_YR):
        if yr in EXCLUDE_YR:
            continue
        try:
            df_to_export = processSingleYr(yr, df_golden, WINDOW_SIZE)
            # export
            df_to_export[
                ['Focal', 'SharedTitle', 'Reference', 'TargetTitle', 
                'Start_Year', 'End_Year', 'Duration', 'Job_Index']
            ].to_csv(os.path.join(export_path, 'graph_etet_{yr}.csv'.format(yr=yr)))
            print('File graph_etet_{yr}.csv is exported'.format(yr=yr), index=False)
        except:
            print('Errored! {yr}'.format(yr=yr))