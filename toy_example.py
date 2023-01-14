import networkx as nx
from src.train import *
from src.dataset import *
from torch_geometric.data import Data
import torch

# create an empty graph
G = nx.DiGraph()

# create the leagues
G.add_node(0, ranking = 1.0) # PL
G.add_node(1, ranking = 2.0) # Bundes
G.add_node(2, ranking = 4.0) # Serie A

# add transfers
G.add_edge(0, 1, age = 25.0, same_country = 0.0)
G.add_edge(2, 0, age = 29.0, same_country = 0.0)
G.add_edge(1, 2, age = 24.0, same_country = 0.0)
G.add_edge(0, 1, age = 26.0, same_country = 0.0)
G.add_edge(2, 1, age = 28.0, same_country = 0.0)
G.add_edge(1, 0, age = 19.0, same_country = 0.0)

adaptability_factor = [0.8, 1.4, 1.6, 1.0, 3.2, 0.25]

dataset = create_dataset(G, adaptability_factor)

train(dataset)


