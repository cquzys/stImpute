import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from scipy.spatial import distance
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_graph_by_gene(X):
    
    Adj = torch.tensor(distance.cdist(X, X, 'cosine')).float()
    return Adj

def pred_genes(net, val_loader, train_lab, scRNA_data, genes_to_predict, n_neighbors=50):
    
    net.eval()
    fm_mu = None
    for _, (x, _) in enumerate(val_loader):
        x = x.cuda()
        encode, _ = net(x)
        if fm_mu is None:
            fm_mu = encode.cpu().detach().numpy()
        else:
            fm_mu = np.concatenate((fm_mu, encode.cpu().detach().numpy()), axis=0)
    
    scRNA_transformed = fm_mu[train_lab!=1,:]
    spatial_transformed = fm_mu[train_lab==1,:]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric = 'cosine').fit(scRNA_transformed)

    pred_res = pd.DataFrame(np.zeros((spatial_transformed.shape[0],genes_to_predict.shape[0])), columns=genes_to_predict)

    distances, indices = nbrs.kneighbors(spatial_transformed)
    for j in range(0,spatial_transformed.shape[0]):
        weights = 1-(distances[j,:][distances[j,:]<1])/(np.sum(distances[j,:][distances[j,:]<1]+1e-12))
        weights = weights/(len(weights)-1)
        pred_res.iloc[j,:] = np.dot(weights, scRNA_data[genes_to_predict].iloc[indices[j,:][distances[j,:] < 1]])
    
    net.train()
    return pred_res

def find_neighbors(net, val_loader, train_lab, scRNA_data, train_gene, test_gene, n_neighbors=50):
    
    net.eval()
    fm_mu = None
    for _, (x, _) in enumerate(val_loader):
        x = x.cuda()
        encode, _ = net(x)
        if fm_mu is None:
            fm_mu = encode.cpu().detach().numpy()
        else:
            fm_mu = np.concatenate((fm_mu, encode.cpu().detach().numpy()), axis=0)
    
    scRNA_transformed = fm_mu[train_lab!=1,:]
    spatial_transformed = fm_mu[train_lab==1,:]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric = 'cosine').fit(scRNA_transformed)
    all_gene = pd.unique(np.hstack((train_gene, test_gene)))
    pred_res = pd.DataFrame(np.zeros((spatial_transformed.shape[0]*n_neighbors, all_gene.shape[0])), columns=all_gene)

    _, indices = nbrs.kneighbors(spatial_transformed)
    
    for j in range(spatial_transformed.shape[0]):
        pred_res.iloc[j*n_neighbors:(j+1)*n_neighbors, :] = scRNA_data[all_gene].iloc[indices[j,:]]
    
    net.train()
    return pred_res

def calc_all(spatial_df, pred_res, cal='cosine'):

    correlation = []
    for i in range(spatial_df.shape[1]):
        if cal=='mse':
            correlation.append(mean_squared_error(spatial_df.iloc[:, i], pred_res.iloc[:, i]))
        elif cal=='cosine':
            correlation.append(1-cosine(spatial_df.iloc[:, i], pred_res.iloc[:, i]))
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