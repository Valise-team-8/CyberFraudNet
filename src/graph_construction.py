import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from utils.logger import log

def build_graph(df, node_features):
    """Build a heterogeneous graph from fraud detection data"""
    log("🔗 Building graph structure...")
    
    G = nx.Graph()
    node_to_idx = {}
    node_labels = {}
    
    # Create nodes and edges
    for idx, row in df.iterrows():
        user = f"user_{row['user_id']}"
        seller = f"seller_{row['seller_id']}" 
        product = f"product_{row['product_id']}"
        
        # Add nodes if not exists
        for node in [user, seller, product]:
            if node not in G.nodes():
                G.add_node(node)
        
        # Add edges with temporal information
        timestamp = row['timestamp'] if 'timestamp' in row else 0
        G.add_edge(user, product, timestamp=timestamp, edge_type='user_product')
        G.add_edge(product, seller, timestamp=timestamp, edge_type='product_seller')
        
        # Store fraud labels for user nodes
        if 'is_fraud' in row:
            node_labels[user] = int(row['is_fraud'])
    
    # Create node mapping
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Create edge index
    edges = []
    for edge in G.edges():
        edges.append([node_to_idx[edge[0]], node_to_idx[edge[1]]])
    
    if edges:
        edge_index = torch.tensor(edges).t().contiguous()
    else:
        # Create a simple edge if no edges exist
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    
    # Prepare node features
    num_nodes = len(nodes)
    feature_dim = node_features.shape[1] if len(node_features.shape) > 1 else node_features.shape[0]
    
    # Assign features to nodes
    if len(node_features) == len(df):
        # Features correspond to dataframe rows, assign to user nodes
        x = torch.zeros(num_nodes, feature_dim, dtype=torch.float)
        
        for idx, row in df.iterrows():
            user = f"user_{row['user_id']}"
            if user in node_to_idx and idx < len(node_features):
                x[node_to_idx[user]] = torch.tensor(node_features[idx], dtype=torch.float)
        
        # Fill remaining nodes with average features
        avg_features = torch.mean(x[x.sum(dim=1) != 0], dim=0)
        for i in range(num_nodes):
            if x[i].sum() == 0:  # Empty feature vector
                x[i] = avg_features
    else:
        # Use features as-is if dimensions match
        if len(node_features) == num_nodes:
            x = torch.tensor(node_features, dtype=torch.float)
        else:
            # Pad or truncate features to match number of nodes
            x = torch.zeros(num_nodes, feature_dim, dtype=torch.float)
            min_len = min(len(node_features), num_nodes)
            x[:min_len] = torch.tensor(node_features[:min_len], dtype=torch.float)
    
    # Create labels tensor
    y = torch.zeros(num_nodes, dtype=torch.long)
    for node, label in node_labels.items():
        if node in node_to_idx:
            y[node_to_idx[node]] = label
    
    # Create data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = num_nodes
    data.num_node_features = feature_dim
    
    log(f"Graph created: {num_nodes} nodes, {edge_index.shape[1]} edges")
    log(f"Node features shape: {x.shape}")
    log(f"Fraud cases in graph: {y.sum().item()}")
    
    return data