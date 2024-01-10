import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader
from utils import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class AutoEncoder(nn.Module):

    def __init__(self, input_dim: int):
        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, input_dim)

    def forward(self, x):
        encode = F.relu(self.fc1(x))
        decode = F.relu(self.fc2(encode))
        return encode, decode
    
class GS_block(nn.Module):

    def __init__(self, input_dim: int=50, output_dim: int=50):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim*2, output_dim))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: Tensor, adj: Tensor):
    
        neigh_feats = self.aggregate(x, adj)
        combined = torch.cat([x.reshape(-1, self.input_dim), neigh_feats.reshape(-1, self.input_dim)], dim=1)
        combined = F.relu(combined @ self.weight)
        combined = F.normalize(combined,2,1).reshape(x.shape[0], -1)
        return combined
        
    def aggregate(self, x: Tensor, adj: Tensor):

        n = len(adj)
        adj = adj-torch.eye(n, device=adj.device)
        adj /= (adj.sum(1, keepdim=True)+1e-12)
        return adj.mm(x)
        
class Trans(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, cell_num: int, hidden_dim: int=256, gnnlayers: int=2, seed: int=42):
        super().__init__()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.n_neighbors = input_dim
        self.mse = nn.MSELoss(reduction='mean')
        self.cos_by_col = nn.CosineSimilarity(dim=1)
        self.cos_by_row = nn.CosineSimilarity(dim=0)

        self.graphlayers = nn.ModuleList([GS_block(input_dim, input_dim) for _ in range(gnnlayers)])
        self.trans = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reliable = nn.Sequential(
            nn.Linear(cell_num, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def train_imp_step(self, X: Tensor, Y: Tensor, graph: Tensor, optimizer):

        X_hat = X.t()
        for layer in self.graphlayers:
            X_hat = layer(X_hat, graph)
        X_hat = X_hat.reshape(-1, self.n_neighbors)
        Y_hat = self.trans(X_hat).reshape(X.shape[1], -1).t()
        
        loss1 = 1 - self.cos_by_col(Y_hat-Y_hat.mean(), Y-Y.mean()).mean()
        loss2 = 1 - self.cos_by_row(Y_hat-Y_hat.mean(), Y-Y.mean()).mean()
        loss3 = self.mse(Y_hat, Y)
        loss = (loss1 + loss2) * 2. + loss3
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def train_reliable_step(self, X: Tensor, Y: Tensor, optimizer):

        reliable_score = self.reliable(X).squeeze()
        loss = self.mse(reliable_score, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    def imp_predict(self, X: Tensor, graph: Tensor):
        
        self.eval()
        with torch.no_grad():
            preds = self(X, graph)
        return preds
    
    def reliable_predict(self, X: Tensor):

        self.eval()
        with torch.no_grad():
            reliable_score = self.reliable(X).squeeze()
        return reliable_score
        
    def forward(self, X: Tensor, graph: Tensor):

        X_hat = X.t()
        for layer in self.graphlayers:
            X_hat = layer(X_hat, graph)
        X_hat = X_hat.reshape(-1, self.n_neighbors)
        Y_hat = self.trans(X_hat).reshape(X.shape[1], -1).t()
        return Y_hat


def scPread(spatial_df, scrna_df, train_gene, test_gene, seed=42, emb_file=None):

    """
    Args:
        spatial_df:     [pandas dataframe] normalized and logarithmized original spatial data (cell by gene)
        scrna_df:       [pandas dataframe] normalized and logarithmized reference scRNA-seq data (cell by gene)
        train_gene:     [numpy array] genes for training
        test_gene:      [numpy array] genes for predicting
        seed:           [int] random seed in torch
        emb_file:       [str] a file containing the results of gene's coding by esm-2 (or None)
    Returns:
        scPread_res:    [pandas dataframe] predicted spatial data (cell by gene)
        reliable_score: [numpy array] predicted gene's reliable score
    """
    
    AE_epochs = 30
    EM_epochs = 2
    M_epochs = 10
    gnnlayers = 2
    
    if spatial_df.shape[0] < 1e4 and spatial_df.shape[1] < 5e2:
        n_neighbors = 50
        E_epochs = 10
        hidden_dim = 256
        alpha = .7
        reliable_epochs = 100
    else:
        n_neighbors = 20
        E_epochs = 50
        hidden_dim = 32
        alpha = .5
        reliable_epochs = 200
        
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    print('ST data:            %d cells * %d genes' % (spatial_df.shape))
    print('scRNA-seq data:     %d cells * %d genes' % (scrna_df.shape))
    print('%d genes to be predicted\n' % (test_gene.shape[0]))

    all_gene = np.hstack((train_gene, test_gene))
    spatial_df = spatial_df[train_gene]
    scrna_df = scrna_df[all_gene]
    
    sorted_spatial_data_label = np.ones(spatial_df.shape[0])
    sorted_scRNA_data_label = np.zeros(scrna_df.shape[0])

    train_data = torch.from_numpy(np.vstack((spatial_df, scrna_df[train_gene]))).float()
    train_label = torch.from_numpy(np.hstack((sorted_spatial_data_label, sorted_scRNA_data_label))).float()

    train_set = TensorsDataset(train_data, train_label)
    train_loader = DataLoader(dataset=train_set, batch_size=512, shuffle=True, num_workers=4, generator=torch.Generator(device='cuda'))
    val_loader = DataLoader(dataset=train_set, batch_size=512, shuffle=False, num_workers=4, generator=torch.Generator(device='cuda'))
    
    ### Training 1: Training AutoEncoder
    AENet = AutoEncoder(train_gene.shape[0]).to(device)
    AENet_optimizer = torch.optim.Adam(AENet.parameters(), lr=1e-2, weight_decay=1e-4)

    pbar = tqdm(range(AE_epochs))
    for _ in pbar:
        for _, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.to(device)
            _, decode = AENet(train_x)
            loss = nn.MSELoss(reduction='mean')(decode[train_y==1], train_x[train_y==1])
            AENet_optimizer.zero_grad()
            loss.backward()
            AENet_optimizer.step()
        pbar.set_description(f'Embedding        ')

    ### Training 2: Training Trans_module
    TransNet = Trans(
                input_dim=n_neighbors,
                output_dim=1,
                cell_num=spatial_df.shape[0],
                hidden_dim=hidden_dim,
                gnnlayers=gnnlayers,
                seed=seed,
                ).to(device)
    TransNet_optimizer = torch.optim.Adam(TransNet.parameters(), lr=1e-2, weight_decay=1e-2) 
    
    if emb_file != None:
        emb = pickle.load(open(emb_file, 'rb'))
        graph = build_graph_by_gene(emb[train_gene].values.T).to(device)
        test_graph = build_graph_by_gene(emb[all_gene].values.T).to(device)
    else:
        graph = build_graph_by_gene(scrna_df[train_gene].values.T).to(device)
        test_graph = build_graph_by_gene(scrna_df[all_gene].values.T).to(device)       
    
    pbar = tqdm(range(EM_epochs))
    for _ in pbar:
    
        new_scrna_df = find_neighbors(AENet, val_loader, train_label, scrna_df, train_gene, test_gene, n_neighbors)
        X = torch.FloatTensor(new_scrna_df[train_gene].values).to(device)
        Y = torch.FloatTensor(spatial_df[train_gene].values).to(device)
        test_X = torch.FloatTensor(new_scrna_df[all_gene].values).to(device)
        raw_X = torch.FloatTensor(spatial_df.values).to(device)

        for _ in range(E_epochs):
            TransNet.train()
            TransNet.train_imp_step(X, Y, graph, TransNet_optimizer)

        for _ in range(M_epochs):
            AENet.train()
            TransNet.eval()
            TransOut = TransNet.imp_predict(X, graph)
            _, AEOut = AENet(raw_X)
            out = (AEOut + TransOut) / 2
            loss = nn.MSELoss(reduction='mean')(out, Y)
            AENet_optimizer.zero_grad()
            loss.backward()
            AENet_optimizer.step()
            
        pbar.set_description(f"EM_training      ")
         
    ### Predicting
    with torch.no_grad():
        TransNet.eval()
        preds_test_X = TransNet.imp_predict(test_X, test_graph)
        mid_ans = pred_genes(AENet, val_loader, train_label, scrna_df, test_gene, n_neighbors)
    
    ### Training 3: Training Reliable_module
    reliable_X = preds_test_X[:, :len(train_gene)]
    reliable_Y = torch.zeros(len(train_gene)).float().to(device)
    reliable_test_X = preds_test_X[:, len(train_gene):]
    for i in range(len(train_gene)):
        reliable_Y[i] = 1-cosine(preds_test_X[:, i].cpu(), spatial_df.iloc[:, i])

    pbar = tqdm(range(reliable_epochs))
    for _ in pbar:
        TransNet.train()
        TransNet.train_reliable_step(reliable_X.T, reliable_Y, TransNet_optimizer)
        pbar.set_description(f"Reliable training")

    ### Reliable score redicting
    reliable_score = TransNet.reliable_predict(reliable_test_X.T).reshape(1, -1).cpu().numpy()
    
    pre = preds_test_X[:, len(train_gene):].cpu().numpy()
    mid = mid_ans / mid_ans.max() * pre.max()
    scPread_res = pre * alpha + mid * (1-alpha)
    
    return scPread_res, reliable_score
