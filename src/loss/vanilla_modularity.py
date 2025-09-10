import torch
import torch.nn as nn

class VanillaModularityLoss(nn.Module):
    def __init__(self, n_clusters, initial_alpha=0.5, initial_beta=2.0, initial_gamma=0.1):
        super(VanillaModularityLoss, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))

    def forward(self, C, X, adj):
        device = C.device
        n = adj.shape[0]
        k = self.n_clusters
        
        # Adjacency matrix tensor
        if not torch.is_tensor(adj):
            adj_tensor = torch.tensor(adj.toarray(), dtype=torch.float32, device=device)
        else:
            adj_tensor = adj.to(device)
            
        deg = torch.sum(adj_tensor, dim=1)
        m = torch.sum(deg) / 2
        d_outer = torch.outer(deg, deg) / (2 * m)
        B = adj_tensor - d_outer
        modularity_term = torch.trace(C.t() @ B @ C)
        mod_loss = -(1 / (2 * m)) * modularity_term

        ones = torch.ones((n, 1), device=device)
        cluster_distribution = C.t() @ ones
        collapse_term = torch.norm(cluster_distribution - 1, p=1)
        collapse_reg = (torch.sqrt(torch.tensor(k, device=device, dtype=torch.float32)) / n) * collapse_term

        eps = 1e-10
        entropy = -torch.sum(C * torch.log(C + eps)) / n

        # Total loss including entropy regularization
        loss = mod_loss + self.beta * collapse_reg
        
        return loss
