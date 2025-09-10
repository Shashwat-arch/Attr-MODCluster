import torch
from kmeans_pytorch import kmeans

def kmeans_clustering(embeddings, num_clusters, device='cpu'):
    """
    Clusters embeddings using kmeans-pytorch and returns cluster assignments.

    Args:
        embeddings (torch.Tensor): 2D tensor of shape [num_points, embedding_dim].
        num_clusters (int): Number of clusters to form.
        device (str): 'cpu' or 'cuda' for computation.

    Returns:
        cluster_ids (torch.Tensor): tensor of cluster indices for each embedding.
        cluster_centers (torch.Tensor): tensor of cluster centers.
    """
    # Ensure embeddings are on the correct device
    embeddings = embeddings.to(device)
    # Perform kmeans clustering
    cluster_ids, cluster_centers = kmeans(
        X=embeddings, num_clusters=num_clusters, distance='euclidean', device=torch.device(device)
    )
    return cluster_ids, cluster_centers
