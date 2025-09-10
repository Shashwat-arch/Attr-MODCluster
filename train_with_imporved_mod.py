import numpy as np
from scipy.sparse import coo_matrix
import torch
import torch.optim as optim
from torch_geometric.utils import subgraph
from src.utils import collect_data, clustering_metrics, mini_batch_sampler, normalize_adj
from src.models.kmeans import kmeans_clustering
from src.models.gcn_and_sage import get_model  # updated import to get_model for GCN or GraphSAGE
from src.loss.improved_modularity_loss import ModularityLoss
from src.plot import plot

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_name = 'Cora'  # example dataset
norm_adj, data, nclasses = collect_data(dataset_name)
features = data.x.to(device)
edge_index = data.edge_index.to(device)
labels = data.y.tolist()

in_dim = features.shape[1]
hidden_dim = 32
dropout = 0.5
num_epochs = 80
batch_size = 64

# Determine if dataset is Planetoid cora or citeseer for full batch training
use_mini_batch = dataset_name not in ['Cora', 'CiteSeer']
plot_path = "E:\PhD\Code\YRS_CODS_25\Rethinking_Modularity_Loss_with_Feature_Information_for_Better_Graph_Clustering\plots"

if use_mini_batch:
    total_dataset_size = norm_adj.shape[0]
    num_batches = total_dataset_size // batch_size
    # batches = mini_batch_sampler(norm_adj, batch_size, num_batches)
    batches = mini_batch_sampler(norm_adj, batch_size, num_batches)
    model_type = 'graphsage'  # Prefer GraphSAGE for mini-batch
else:
    model_type = 'gcn'  # Full-batch for GCN

model = get_model(model_type, in_dim, hidden_dim, nclasses, dropout).to(device)
loss_fn = ModularityLoss(nclasses, initial_alpha=0.5).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(loss_fn.parameters()),
    lr=1e-3
)

# Convert norm_adj to tensor on device if not already
if not torch.is_tensor(norm_adj):
    adj_norm = torch.tensor(norm_adj.toarray(), dtype=torch.float32, device=device)
else:
    adj_norm = norm_adj.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    if use_mini_batch:
        batch_num = 0
        for batch_nodes in batches:
            batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long, device=device)
            batch_edge_index, _ = subgraph(batch_nodes_tensor, edge_index, relabel_nodes=True)
            batch_features = features[batch_nodes_tensor]

            rows = batch_edge_index[0].cpu().numpy()
            cols = batch_edge_index[1].cpu().numpy()
            data_arr = np.ones(len(rows), dtype=np.float32)
            batch_adj = coo_matrix((data_arr, (rows, cols)), shape=(len(batch_nodes), len(batch_nodes)))

            batch_norm_adj = normalize_adj(batch_adj).tocoo()
            indices = torch.vstack([torch.tensor(batch_norm_adj.row), torch.tensor(batch_norm_adj.col)]).to(device)
            values = torch.tensor(batch_norm_adj.data).to(device)
            adj_tensor = torch.sparse_coo_tensor(indices, values, batch_norm_adj.shape, device=device)

            optimizer.zero_grad()
            C = model(batch_features, batch_edge_index)
            loss = loss_fn(C, batch_features, adj_tensor)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            total_loss += loss.item()
            batch_num += 1
        avg_loss = total_loss / num_batches
    else:
        # Full-batch training for Cora, Citeseer with GCN
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = loss_fn(out, features, adj_norm)
        loss.backward()
        optimizer.step()
        avg_loss = loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Avg Loss={avg_loss}, Alpha={loss_fn.alpha.item()}, Beta={loss_fn.beta.item()}, Gamma={loss_fn.gamma.item()}")

model.eval()
with torch.no_grad():
    out = model(features, edge_index)
    pred_labels = out.argmax(dim=1).cpu().numpy()

true_classes = set(labels if not torch.is_tensor(labels) else labels.cpu().numpy())
pred_classes = set(pred_labels if not torch.is_tensor(pred_labels) else pred_labels.cpu().numpy())
print("True classes:", true_classes)
print("Predicted classes:", pred_classes)

print("Plotting started!!")
plot(out, labels, plot_path, dataset_name)

acc, nmi, ari, f1_macro = clustering_metrics(labels, pred_labels)
print(f"Accuracy: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, F1 Macro: {f1_macro:.4f}")

# # k-means clustering on logits as embeddings
# cluster_ids, cluster_centers = kmeans_clustering(out, nclasses, device)
# acc, nmi, ari, f1_macro = clustering_metrics(labels, cluster_ids.cpu().numpy())
# print(acc, nmi, ari, f1_macro)
