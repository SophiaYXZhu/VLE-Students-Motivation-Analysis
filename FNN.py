from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, n_classes_in, n_classes_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_classes_in, 128), 
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes_out), 
            # nn.Softmax()
        )

    def forward(self, X):
        return self.net(X)
