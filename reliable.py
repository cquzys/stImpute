import torch
import torch.nn as nn
from scipy.spatial.distance import cosine

def cal_reliable(spatial_df: torch.FloatTensor, reliable_X: torch.FloatTensor, reliable_test_X: torch.FloatTensor):

    model = nn.Sequential(
        nn.Linear(reliable_X.shape[0], 256*2),
        nn.ReLU(),
        nn.Linear(256*2, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2) 
    
    reliable_Y = torch.zeros(reliable_X.shape[1]).float()
    for i in range(reliable_X.shape[1]):
        reliable_Y[i] = 1-cosine(reliable_X[:, i].cpu(), spatial_df[:, i].cpu())

    for _ in range(200):
        reliable_score = model(reliable_X.T).squeeze()
        loss = nn.MSELoss(reduction='mean')(reliable_score, reliable_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        reliable_score = model(reliable_test_X.T).squeeze().reshape(1, -1)
    return reliable_score.cpu()