import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from model import *

seed=42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
set_seed(seed)

torch.set_num_threads(10)

spatial_df_file = 'dataset/osmFISH.pkl'
scrna_df_file   = 'dataset/Zeisel.pkl'
raw_spatial_df  = pickle.load(open(spatial_df_file, 'rb'))
raw_scrna_df    = pickle.load(open(scrna_df_file, 'rb'))
raw_scrna_df[np.isnan(raw_scrna_df)] = 0
raw_shared_gene = np.intersect1d(raw_spatial_df.columns, raw_scrna_df.columns)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
kf.get_n_splits(raw_shared_gene)
idx = 1
for train_ind, test_ind in kf.split(raw_shared_gene):    
    print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d"%(idx, len(train_ind), len(test_ind)))
    train_gene = raw_shared_gene[train_ind]
    test_gene  = raw_shared_gene[test_ind]
    test_spatial_df = raw_spatial_df[test_gene]
    spatial_df = raw_spatial_df
    scrna_df   = raw_scrna_df
    
    if idx == 1:
        all_pred_res = pd.DataFrame(np.zeros((spatial_df.shape[0],raw_shared_gene.shape[0])), columns=raw_shared_gene)
    if not os.path.exists('./result'):
        os.makedirs('./result') 
    save_path_prefix = './result/fold%d'%(idx)
    all_pred_res[test_gene] = Model(spatial_df, scrna_df, train_gene, test_gene, save_path_prefix, n_neighbors=50 if spatial_df.shape[0] < 1e4 else 20, seed=seed)
    idx += 1


print('\nresult: ')
print('gene-wise cosine: %.3f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene], all_pred_res, cal='cosine'))))
#print('gene-wise pearsonr: %.3f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene], all_pred_res, cal='pearsonr'))))
#print('gene-wise spearmanr: %.3f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene], all_pred_res, cal='spearmanr'))))
print('gene-wise mse: %.3f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene], all_pred_res, cal='mse'))))
print('cell-wise cosine: %.3f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene].T, all_pred_res.T, cal='cosine'))))
#print('cell-wise pearsonr: %.3f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene].T, all_pred_res.T, cal='pearsonr'))))
#print('cell-wise spearmanr: %.3f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene].T, all_pred_res.T, cal='spearmanr'))))
print('cell-wise mse: %.3f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene].T, all_pred_res.T, cal='mse'))))

#pickle.dump(raw_spatial_df[raw_shared_gene], open('writepicture/osmFISH_raw_spatial_df.pkl', 'wb'))
#pickle.dump(all_pred_res, open('writepicture/osmFISH_pred.pkl', 'wb'))
