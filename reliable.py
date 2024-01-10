import torch
import torch.nn as nn
from torch import Tensor
from scipy.spatial.distance import cosine

def cal_reliable(spatial_df: Tensor, reliable_X: Tensor, reliable_test_X: Tensor):

    """
    Args:
        spatial_df:       [torch.Tensor] normalized and logarithmized original spatial data (cell by gene)
        reliable_X:       [torch.Tensor] predicted results of the model for train genes (cell by gene)
        reliable_test_X:  [torch.Tensor] predicted results of the model for test genes (cell by gene)
    Returns:
        reliable_score:   [torch.Tensor] predicted test gene's reliable score
    """
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
