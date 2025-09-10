import torch
import torch.nn as nn

class ModularityLoss(nn.Module):
    def __init__(self, n_clusters, initial_alpha=0.3, initial_beta=2.0, initial_gamma=0.1):
        super(ModularityLoss, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))  # entropy regularization weight

    def forward(self, C, X, adj):
        device = C.device
        n = adj.shape[0]
        k = self.n_clusters
        
        # Convert sparse adjacency to dense for operations
        if torch.is_tensor(adj) and adj.is_sparse:
            adj_tensor = adj.to_dense()
        elif not torch.is_tensor(adj):
            adj_tensor = torch.tensor(adj.toarray(), dtype=torch.float32, device=device)
        else:
            adj_tensor = adj.to(device)
            
        deg = torch.sum(adj_tensor, dim=1)
        m = torch.sum(deg) / 2
        d_outer = torch.outer(deg, deg) / (2 * m)
        B = adj_tensor - d_outer
        modularity_term = torch.trace(C.t() @ B @ C)
        mod_loss = -(1 / (2 * m)) * modularity_term
    
        # Rest of your code unchanged
        x_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
        W = torch.mm(x_norm, x_norm.t()).clamp(min=0)
        s = torch.sum(W, dim=1)
        w = torch.sum(s) / 2
        s_outer = torch.outer(s, s) / (2 * w + 1e-8)
        B_attr = W - s_outer
        attr_mod_term = torch.trace(C.t() @ B_attr @ C)
        attr_loss = -(1 / (2 * w + 1e-8)) * attr_mod_term
        ones = torch.ones((n, 1), device=device)
        cluster_distribution = C.t() @ ones
        collapse_term = torch.norm(cluster_distribution - 1, p=1)
        collapse_reg = (torch.sqrt(torch.tensor((k * k), device=device, dtype=torch.float32)) / n) * collapse_term
        eps = 1e-10
        entropy = -torch.sum(C * torch.log(C + eps)) / n
        loss = (self.alpha * mod_loss) + (1 - self.alpha) * attr_loss + self.beta * collapse_reg - self.gamma * entropy
        
        return loss

