import torch
import torch.optim as optim
from src.utils import collect_data, clustering_metrics
from src.models.kmeans import kmeans_clustering
from src.models.gcnmodel import GCNModel     # Now Encoder includes loss functionality
from src.loss.vanilla_modularity import VanillaModularityLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
norm_adj, data, nclasses = collect_data('Cora')
features = data.x.to(device)
edge_index = data.edge_index.to(device)
labels = data.y.tolist()

in_dim = features.shape[1]
hidden_dim = 64  # or desired size
dropout = 0.5
num_epochs = 100

gcn_model = GCNModel(in_dim, hidden_dim, nclasses, dropout).to(device)
loss_fn = VanillaModularityLoss(nclasses).to(device)

optimizer = torch.optim.Adam(
    list(gcn_model.parameters()) + list(loss_fn.parameters()),
    lr=1e-3
)

# Ensure norm_adj is tensor and on device
if not torch.is_tensor(norm_adj):
    adj_norm = torch.tensor(norm_adj.toarray(), dtype=torch.float32, device=device)
else:
    adj_norm = norm_adj.to(device)

for epoch in range(num_epochs):
    gcn_model.train()
    optimizer.zero_grad()
    C = gcn_model(features, edge_index)
    loss = loss_fn(C, features, adj_norm)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}: Loss={loss.item()}')

gcn_model.eval()
with torch.no_grad():
    C = gcn_model(features, edge_index)
    pred_labels = C.argmax(dim=1).cpu().numpy()

true_classes = set(labels.cpu().numpy() if torch.is_tensor(labels) else labels)
pred_classes = set(pred_labels.cpu().numpy() if torch.is_tensor(pred_labels) else pred_labels)

print("True classes:", true_classes)
print("Predicted classes:", pred_classes)

# Compute clustering performance metrics
acc, nmi, ari, f1_macro = clustering_metrics(labels, pred_labels)
print(f"Accuracy: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, F1 Macro: {f1_macro:.4f}")

# k-means clustering on embeddings (logits as embeddings)
cluster_ids, cluster_centers = kmeans_clustering(logits, nclasses, device)
acc, nmi, ari, f1_macro = clustering_metrics(labels, cluster_ids.cpu().numpy())
print(acc, nmi, ari, f1_macro)
