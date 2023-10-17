import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import Tensor, nn
from scipy.spatial import distance
from torch.utils.data import DataLoader
from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class AutoEncoder(nn.Module):

    def __init__(self, n_features: int,):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(n_features, 1000)
        self.fc2 = nn.Linear(1000, n_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return h1

    def decode(self, z):
        h2 = F.relu(self.fc2(z))
        return h2

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
    
class GS_block(nn.Module):

    def __init__(self, input_dim: int=50, output_dim: int=50):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(2*self.input_dim, output_dim))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x, adj):
    
        neigh_feats = self.aggregate(x, adj)

        '''
        combined = torch.cat([x, neigh_feats], dim=1)
        combined = F.relu(combined @ self.weight)
        combined = F.normalize(combined,2,1)
        '''
        combined = torch.cat([x.reshape(-1, self.input_dim), neigh_feats.reshape(-1, self.input_dim)], dim=1).reshape(x.shape[0], -1)
        combined = F.relu(combined.reshape(-1, 2*self.input_dim) @ self.weight)
        combined = F.normalize(combined,2,1).reshape(x.shape[0], -1)
        
        return combined
        
    def aggregate(self, x, adj):

        n = len(adj)
        adj = adj-torch.eye(n, device=adj.device)
        adj /= (adj.sum(1, keepdim=True)+1e-12)
        return adj.mm(x)
        

class Trans(nn.Module):

    def __init__(self, dim_tgt_outputs: int, dim_ref_inputs: int, hidden_dim: int=256, 
                 n_neighbors: int=50, gnnlayers: int=2, seed: int=42,
                ):
        super().__init__()
        self.seed = seed
        if not seed is None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.n_neighbors = n_neighbors
        self.relu = nn.ReLU()

        self.graphlayers = nn.ModuleList([GS_block(n_neighbors, n_neighbors) for _ in range(gnnlayers)])
        self.trans = nn.Sequential(
            nn.Linear(dim_ref_inputs, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_tgt_outputs),
        )
        
    def predict(self, X: Tensor, graph: Tensor) -> np.ndarray:
        
        self.eval()
        with torch.no_grad():
            preds = self(X, graph)
        return preds.cpu().numpy()

    def train_imp_step(self, X: Tensor, Y: Tensor, graph: Tensor, optimizer):

        '''
        Y_hat = torch.zeros_like(Y).float()
        for i in range(Y.shape[0]):
            X_hat = X[i*self.n_neighbors:(i+1)*self.n_neighbors, :].t().to('cuda')
            for layer in self.graphlayers:
                X_hat = layer(X_hat, graph)
            Y_hat[i, :] = self.transform(X_hat).t()
        '''
        X_hat = X.t()
        for layer in self.graphlayers:
            X_hat = layer(X_hat, graph)
        X_hat = X_hat.reshape(-1, self.n_neighbors)
        Y_hat = self.trans(X_hat).reshape(-1, Y.shape[0]).t()
        
        loss = self.mse(Y_hat, Y)
        item = loss.item()        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return item
        
    def forward(self, X: Tensor, graph: Tensor):

        X_hat = X.t()
        for layer in self.graphlayers:
            X_hat = layer(X_hat, graph)
        X_hat = X_hat.reshape(-1, self.n_neighbors)
        Y_hat = self.trans(X_hat).reshape(X.shape[1], -1).t()
        return Y_hat


def Model(spatial_df, scrna_df, train_gene, test_gene, save_path_prefix='./stPlus',
          top_k=2000, t_min=5, seed=42, data_quality=None, n_neighbors=50,
           converge_ratio=0.004, max_epoch_num=30, batch_size=512, learning_rate=None, weight_decay=0.0002):
    
    print('Models will be saved in: %s-%dmin.pt\n'%(save_path_prefix, t_min))
    print('Spatial transcriptomics data: %d cells * %d genes'%(spatial_df.shape))
    print('Reference scRNA-seq data:     %d cells * %d genes'%(scrna_df.shape))
    print('%d genes to be predicted\n'%(test_gene.shape[0]))
    print('Start initialization')
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    train_gene = np.array(train_gene)
    test_gene = np.array(test_gene)
    all_gene = np.hstack((train_gene, test_gene))
    spatial_df_all = spatial_df[all_gene]
    spatial_df = spatial_df[train_gene]
    
    # build graph
    def build_graph_by_gene(X, k):
        nodes = X.shape[0]
        Adj = torch.zeros((nodes,nodes))
        pbar = tqdm(range(X.shape[0]))
        for i in pbar:
            distMat = distance.cdist(X[i, :].reshape(1,-1), X, 'cosine')
            res = distMat.argsort()[:k+1]
            for j in np.arange(0,k+1):
                Adj[i][res[0][j]]=1.0
            pbar.set_description(f'Graph_building   ')
        return Adj
    
    print('build graph...')
    graph = build_graph_by_gene(scrna_df[train_gene].values.T, 5).to(device)
    test_graph = build_graph_by_gene(scrna_df[all_gene].values.T, 5).to(device)
    
    shared_gene = np.intersect1d(spatial_df.columns, scrna_df.columns)
    reserved_gene = np.hstack((shared_gene, test_gene))
    
    spatial_df = spatial_df[shared_gene]
    raw_scrna_uniq_gene = np.unique(scrna_df.columns.values[~np.isin(scrna_df.columns.values, reserved_gene)])
    scrna_df = scrna_df[np.hstack((reserved_gene, raw_scrna_uniq_gene))]
    
    spatial_df_appended = np.hstack((spatial_df.values, 
           np.zeros((spatial_df.shape[0], scrna_df.shape[1]-spatial_df.shape[1]))))
    spatial_df_appended = pd.DataFrame(data=spatial_df_appended,
                               index = spatial_df.index, columns=scrna_df.columns)
    
    t_min_loss = 1e9
    
    # select gene
    dedup_ind = ~scrna_df.columns.duplicated()
    spatial_df_appended = spatial_df_appended.loc[:,dedup_ind]
    scrna_df = scrna_df.loc[:,dedup_ind]
    
    other_genes = np.setdiff1d(scrna_df.columns.values, reserved_gene)
    other_genes_mtx = scrna_df[other_genes].values
    selected_ind = select_top_variable_genes(other_genes_mtx, top_k)
    selected_gene = other_genes[selected_ind]
    new_genes = np.hstack((shared_gene, test_gene, selected_gene))
    spatial_df_appended = spatial_df_appended[new_genes]
    scrna_df = scrna_df[new_genes]

    sorted_spatial_data_label = np.ones(spatial_df_appended.shape[0])
    sorted_scRNA_data_label = np.zeros(scrna_df.shape[0])

    train_dat = torch.from_numpy(np.vstack((spatial_df_appended, scrna_df))).float()
    train_lab = torch.from_numpy(np.hstack((sorted_spatial_data_label, sorted_scRNA_data_label))).float()

    net = AutoEncoder(train_dat.shape[1]).cuda()
    if learning_rate is None:
        learning_rate = 4e-3 if scrna_df.shape[0]<1e4 else 8e-5
        
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay=weight_decay)

    train_set = TensorsDataset(train_dat,train_lab)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4, generator=torch.Generator(device = 'cuda'))
    val_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4, generator=torch.Generator(device = 'cuda'))
    if data_quality is None:
        data_quality = 1 - np.sum(np.sum(scrna_df==0)) / (scrna_df.shape[0]*scrna_df.shape[1])
        
    loss_last = 0
    pbar = tqdm(range(max_epoch_num))
    for e in pbar:
        train_loss = 0
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.cuda()
            is_spatial = train_y==1
            pre, _ = net(train_x)

            loss_recon_target = loss_recon_func(pre[is_spatial,:shared_gene.shape[0]], train_x[is_spatial,:shared_gene.shape[0]])

            train_x_new = train_x[~is_spatial]
            train_x_new2 = train_x[~is_spatial]
            train_x_new2[:,shared_gene.shape[0]:] = 0
            decode_output,_ = net(train_x_new2)
            pred = decode_output[:,shared_gene.shape[0]:]
            gt = train_x_new[:,shared_gene.shape[0]:]
            loss_cor_source = loss_recon_sparsity_func(pred,gt,data_quality) * shared_gene.shape[0] / (train_x.shape[1]-shared_gene.shape[0]) * spatial_df_appended.shape[0] / scrna_df.shape[0] 

            loss = loss_recon_target + loss_cor_source

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / (batch_idx + 1)
        if e == 0:
            loss_last = train_loss
        else:
            loss_last = train_loss
        pbar.set_description(f'Embedding        ')
        
        # if e > 0 and ratio < converge_ratio: break

        if train_loss < t_min_loss:
            t_min_loss = train_loss
            torch.save({'epoch': e,'model_state_dict': net.state_dict(),'loss': train_loss,
                        'optimizer_state_dict': optimizer.state_dict()}, '%s-%dmin.pt'%(save_path_prefix,t_min))

    checkpoint = torch.load('%s-%dmin.pt'%(save_path_prefix,t_min))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # CHANGE
    scrna_df = pred_genes(net, val_loader, train_lab, scrna_df, train_gene, test_gene, n_neighbors)

    raw_scrna_df = scrna_df
    raw_spatial_df = spatial_df_all

    epochs = 20 if spatial_df.shape[0] < 1e4 else 100

    X = raw_scrna_df[train_gene].values
    Y = raw_spatial_df[train_gene].values
    test_X = raw_scrna_df[all_gene].values

    #selected_ind = select_top_variable_genes(X, 1000)
    #graph_X = torch.FloatTensor(X[:, selected_ind]).to(device)
    
    # is class by gene
    X = torch.FloatTensor(X).to(device)
    Y = torch.FloatTensor(Y).to(device)
    test_X = torch.FloatTensor(test_X).to(device)
    
    model = Trans(
                dim_tgt_outputs=1,
                dim_ref_inputs=n_neighbors,
                hidden_dim=256 if spatial_df.shape[0] < 1e4 else 32,
                n_neighbors=n_neighbors,
                gnnlayers=2, # if spatial_df.shape[0] < 1e4 else 1,
                seed=seed,
                ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)        
    pbar = tqdm(range(epochs))

    for ith_epoch in pbar:
        model.train()
        loss = model.train_imp_step(X, Y, graph, optimizer)
        if ith_epoch == 0:
            ratio = 1
            loss_last = loss
        else:
            ratio = np.abs(loss_last - loss) / loss_last
            loss_last = loss
            
        pbar.set_description(f"Predicting       ")
        '''
        if ratio < 0.001: 
            print('early stop')
            break
        '''
        
    with torch.no_grad():
        model.eval()
        #preds_X = model.predict(X, graph)
        preds_test_X = model.predict(test_X, test_graph)
    
    return preds_test_X[:, len(train_gene):]


