from torch_geometric.data import Data
import torch

class SoccerDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return self.data
    
    def __len__(self):
        return 1

def create_dataset(G, adaptability_factor):
    # Node feature matrix with shape [num_nodes, num_node_features]
    x = torch.tensor([list(G.nodes[node].values()) for node in G.nodes()], dtype = torch.float32)
    
    # Graph connectivity with shape [2, num_edges]
    edge_index = torch.tensor([edge for edge in G.edges()], dtype = torch.long).T

    # Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor([list(G.edges[edge].values()) for edge in G.edges()], dtype = torch.float32)

    # ground-truth labels
    y = torch.tensor(adaptability_factor, dtype = torch.float32)

    data = Data(
        x = x,
        edge_index = edge_index,
        edge_attr = edge_attr,
        y = y
    )

    return SoccerDataset(data)