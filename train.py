import torch
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from sklearn import metrics
from sklearn.model_selection import KFold
from scipy.spatial.distance import cosine
from model import *
warnings.filterwarnings('ignore')

seed=42
set_seed(seed)

### Loading data
st_adata = sc.read_h5ad('dataset/st-seq/osmFISH.h5ad')
sc_adata = sc.read_h5ad('dataset/scRNA-seq/Zeisel.h5ad')
emb_file = 'embed/osmFISH_emb.pkl' # emb_file = None

raw_spatial_df = pd.DataFrame(st_adata.X, columns=st_adata.var_names)
raw_scrna_df = pd.DataFrame(sc_adata.X, columns=sc_adata.var_names)
raw_shared_gene = np.intersect1d(raw_spatial_df.columns, raw_scrna_df.columns)

all_pred_res = pd.DataFrame(np.zeros((raw_spatial_df.shape[0], raw_shared_gene.shape[0])), columns=raw_shared_gene)
all_reliable_res = pd.DataFrame(np.zeros((1, raw_shared_gene.shape[0])), columns=raw_shared_gene)

### 5-fold cross validation
idx = 1
kf = KFold(n_splits=5, shuffle=True, random_state=0)
kf.get_n_splits(raw_shared_gene)
for train_ind, test_ind in kf.split(raw_shared_gene):    
    print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (idx, len(train_ind), len(test_ind)))
    train_gene = raw_shared_gene[train_ind]
    test_gene  = raw_shared_gene[test_ind]
    spatial_df = raw_spatial_df[train_gene]
    scrna_df   = raw_scrna_df
    all_pred_res[test_gene], all_reliable_res[test_gene] = scPread(spatial_df, scrna_df, train_gene, test_gene, seed=seed, emb_file=emb_file)
    idx += 1
    
pickle.dump(all_pred_res, open('scPread.pkl', 'wb'))

### Calculating metrics
print('result: ')
print('gene-wise cosine: %.2f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene], all_pred_res, cal='cosine'))))
print('gene-wise mse: %.2f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene], all_pred_res, cal='mse'))))
print('cell-wise cosine: %.2f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene].T, all_pred_res.T, cal='cosine'))))
print('cell-wise mse: %.2f' % (np.median(calc_all(raw_spatial_df[raw_shared_gene].T, all_pred_res.T, cal='mse'))))

top50_ind = np.argsort(-all_reliable_res.values.squeeze())[:raw_shared_gene.shape[0]//2]
top50_gene_cos = [1-cosine(all_pred_res[gene], raw_spatial_df[gene]) for gene in raw_shared_gene[top50_ind]]
print('top50%% gene-wise cosine: %.2f' % (np.median(top50_gene_cos)))

imp_adata = ad.AnnData(all_pred_res)
try:
    imp_adata.obs['Cluster'] = st_adata.obs['Cluster'].values.astype('category')
    sc.pp.neighbors(imp_adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(imp_adata)
    sc.tl.leiden(imp_adata)

    labels_true = []
    dic = dict()
    for str in st_adata.obs['Cluster']:
        try:
            labels_true.append(dic[str])
        except:
            dic[str] = len(dic)
            labels_true.append(dic[str])
    labels_true = np.array(labels_true)
    labels_pred = imp_adata.obs['leiden'].astype('int').to_numpy()

    print('ARI: %.2f' % (metrics.adjusted_rand_score(labels_true, labels_pred)))
    print('FMI: %.2f' % (metrics.fowlkes_mallows_score(labels_true, labels_pred)))
    print('Comp: %.2f' % (metrics.completeness_score(labels_true, labels_pred)))
except:
    exit()
