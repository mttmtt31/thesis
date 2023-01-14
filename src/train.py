import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from src.model import GNN

def train(soccer_dataset):
    input_dim = soccer_dataset.data.x.shape[1]
    hidden_dim = 8
    output_dim = soccer_dataset.data.y.shape[0]
    
    # Define the GNN model
    model = GNN(input_dim, hidden_dim, output_dim)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Define a DataLoader to handle the soccer_dataset
    loader = DataLoader(soccer_dataset, batch_size=1, shuffle=True)

    # Training loop
    for epoch in range(100):
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index)
            loss = loss_fn(output, batch.y)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}")
    return model


