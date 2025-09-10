import numpy as np
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from scipy.sparse import coo_array, coo_matrix
from sklearn import metrics
from munkres import Munkres

import numpy as np
import scipy.sparse as sp

def mini_batch_sampler(adj, batch_size, num_batches, 
                                walk_length=4, num_walks=20):
    """
    Efficient mini-batch sampling of nodes using random walk neighborhood expansion.
    - adj: normalized adjacency matrix (scipy sparse csr matrix)
    - batch_size: number of nodes per batch
    - num_batches: number of batches to sample
    - walk_length: number of steps per walk (default 4)
    - num_walks: number of walks per node (default 20)
    Returns a list of numpy arrays for each batch, each array containing the batch nodes + expanded neighbors.
    """
    nnodes = adj.shape[0]
    all_nodes = np.arange(nnodes)
    batches = []
    for _ in range(num_batches):
        # 1. Sample batch nodes
        batch_nodes = np.random.choice(all_nodes, size=batch_size, replace=False)
        # 2. Expand neighborhood by random walks
        visited = set(batch_nodes)
        for node in batch_nodes:
            for _ in range(num_walks):
                curr = node
                for _ in range(walk_length):
                    neighbors = adj[curr].nonzero()[1]
                    # print("neighbors", neighbors, neighbors[1])
                    if len(neighbors) == 0:
                        break
                    next_node = np.random.choice(neighbors)
                    visited.add(next_node)
                    curr = next_node
        subgraph_nodes = np.array(list(visited))
        print("Nodes in current batch: ", subgraph_nodes)
        batches.append(subgraph_nodes)
    return batches


def normalize_adj(adj):
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = coo_matrix(np.diag(d_inv_sqrt))
    return D_inv_sqrt @ adj @ D_inv_sqrt

def collect_data(dataset_name):
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='E:/PhD/Code/Dataset', name=dataset_name)
    elif dataset_name in ['Computers', 'Photo', 'Amazon_PC', 'Amazon_Computers']:
        # Considering "Amazon_PC" and "Amazon_Computers" as variants of Amazon dataset
        # Torch Geometric has Amazon dataset for Computers and Photo
        # Adjust dataset_name to 'Computers' or 'Photo' if needed accordingly
        if dataset_name == 'Amazon_PC':
            dataset = Amazon(root='E:/PhD/Code/Dataset', name='Computers')  # Map PC to Computers
        elif dataset_name == 'Amazon_Computers':
            dataset = Amazon(root='E:/PhD/Code/Dataset', name='Computers')
        else:
            dataset = Amazon(root='E:/PhD/Code/Dataset', name=dataset_name)
    elif dataset_name in ['Coauthor_CS', 'Coauthor_PHY']:
        if dataset_name == 'Coauthor_CS':
            dataset = Coauthor(root='E:/PhD/Code/Dataset', name='CS')
        else:  # Coauthor_PHY
            dataset = Coauthor(root='E:/PhD/Code/Dataset', name='Physics')
    else:
        raise ValueError(f"Dataset {dataset_name} is not recognized for loading.")
    
    data = dataset[0]
    nclasses = dataset.num_classes
    print("Dataset downloaded successfully!")
    nnodes = data.x.shape[0]
    nfeatures = data.x.shape[1]
    nedges = data.edge_index.shape[1]
    row, col = data.edge_index
    weights = np.ones(len(row))
    row, col = row.numpy(), col.numpy()
    adj = coo_array((weights, (row, col)), shape=[nnodes, nnodes])
    adj = adj.tocsr()
    norm_adj = normalize_adj(adj)
    return norm_adj, data, nclasses


def clustering_metrics(true_labels, pred_labels):
    # Best mapping between true_labels and pred_labels
    l1 = list(set(true_labels))
    l2 = list(set(pred_labels))
    numclass1 = len(l1)
    numclass2 = len(l2)
    if numclass1 != numclass2:
        raise ValueError("Number of classes in true and predicted labels do not match!")
    # Build cost matrix
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_labels) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_labels[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    # Remap pred_labels
    new_pred = np.zeros(len(pred_labels))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(pred_labels) if elm == c2]
        new_pred[ai] = c
    acc = metrics.accuracy_score(true_labels, new_pred)
    f1_macro = metrics.f1_score(true_labels, new_pred, average='macro')
    nmi = metrics.normalized_mutual_info_score(true_labels, pred_labels)
    ari = metrics.adjusted_rand_score(true_labels, pred_labels)
    return acc, nmi, ari, f1_macro