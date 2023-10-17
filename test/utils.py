import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.stats as st
import torchvision.transforms as transforms
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset

def pred_genes(net, val_loader, train_lab, scRNA_data, train_gene, test_gene, n_neighbors=50):
    
    net.eval()
    fm_mu = None
    for batch_idx, (x, _) in enumerate(val_loader):
        x = x.cuda()
        decode_output, mu = net(x)
        # mu = x
        if fm_mu is None:
            fm_mu = mu.cpu().detach().numpy()
        else:
            fm_mu = np.concatenate((fm_mu,mu.cpu().detach().numpy()),axis=0)
    
    scRNA_transformed = fm_mu[train_lab!=1,:]
    spatial_transformed = fm_mu[train_lab==1,:]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric = 'cosine').fit(scRNA_transformed)
    all_gene = np.hstack((train_gene, test_gene))
    pred_res = pd.DataFrame(np.zeros((spatial_transformed.shape[0]*n_neighbors, all_gene.shape[0])), columns=all_gene)

    distances, indices = nbrs.kneighbors(spatial_transformed)
    
    pbar = tqdm(range(spatial_transformed.shape[0]))
    for j in pbar:
        weights = 1-(distances[j,:])/(np.sum(distances[j,:]))
        weights = weights/(len(weights)-1)
        # pred_res.iloc[j,:] = np.dot(weights,scRNA_data[genes_to_predict].iloc[indices[j,:][distances[j,:] < 1]])
        pred_res.iloc[j*n_neighbors:(j+1)*n_neighbors, :] = scRNA_data[all_gene].iloc[indices[j,:]]
        pbar.set_description(f'Neighbor_finding ')
    
    net.train()
    return pred_res

def select_top_variable_genes(data_mtx, top_k):
    """
    data_mtx: data matrix (cell by gene)
    top_k: number of highly variable genes to choose
    """
    var = np.var(data_mtx, axis=0)
    ind = np.argpartition(var,-top_k)[-top_k:]
    return ind

def calc_corr(spatial_df, pred_res, test_gene, cal='cosine'):
    """
    spatial_df: original spatial data (cell by gene dataframe)
    pred_res: predicted results (cell by gene dataframe)
    test_gene: genes to calculate Spearman correlation
    """
    correlation = []
    for gene in test_gene:
        if cal=='cosine':
            correlation.append(1-cosine(spatial_df[gene], pred_res[gene]))
        elif cal=='pearsonr':
            correlation.append(st.pearsonr(spatial_df[gene], pred_res[gene])[0])
        else:
            correlation.append(st.spearmanr(spatial_df[gene], pred_res[gene])[0])
    return correlation
    
def calc_all(spatial_df, pred_res, cal='cosine'):
    """
    spatial_df: original spatial data (cell by gene dataframe)
    pred_res: predicted results (cell by gene dataframe)
    test_gene: genes to calculate Spearman correlation
    """
    correlation = []
    for i in range(spatial_df.shape[1]):
        if cal=='mse':
            correlation.append(mean_squared_error(spatial_df.iloc[:, i], pred_res.iloc[:, i]))
        elif cal=='cosine':
            correlation.append(1-cosine(spatial_df.iloc[:, i], pred_res.iloc[:, i]))
        elif cal=='pearsonr':
            correlation.append(st.pearsonr(spatial_df.iloc[:, i], pred_res.iloc[:, i])[0])
        else:
            correlation.append(st.spearmanr(spatial_df.iloc[:, i], pred_res.iloc[:, i])[0])
    return correlation

class TensorsDataset(TensorDataset):

    def __init__(self, data, target=None, transforms=None, target_transforms=None):
        if target is not None:
            assert data.size(0) == target.size(0) 
        self.data = data
        self.target = target
        if transforms is None:                    
            transforms = []
        if target_transforms is None:         
            target_transforms = []
        if not isinstance(transforms, list):              
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):                    
        data = self.data[index]             
        for transform in self.transforms:               
            data = transform(data)
        if self.target is None:                    
            return data
        target = self.target[index]                   
        for transform in self.target_transforms:           
            target = transforms.ToTensor(transform(transforms.ToPILImage(target)))
        return (data, target)

    def __len__(self):                           
        return self.data.size(0)
    
def loss_recon_func(recon_x, origi_x):

    MSE = nn.MSELoss(reduction='sum')(recon_x, origi_x)
    return MSE

def loss_recon_sparsity_func(recon_x, origi_x, data_quality):
    
    zero_ind = origi_x==0
    non_zero_ind = ~zero_ind
    recon_x_0 = recon_x[zero_ind]
    recon_x_1 = recon_x[non_zero_ind]
    origi_x_0 = origi_x[zero_ind]
    origi_x_1 = origi_x[non_zero_ind]
    MSE_0 = nn.MSELoss(reduction='sum')(recon_x_0, origi_x_0)
    MSE_1 = nn.MSELoss(reduction='sum')(recon_x_1, origi_x_1)
    MSE = data_quality * MSE_0 + MSE_1
    return MSE