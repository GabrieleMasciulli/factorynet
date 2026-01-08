import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import zarr
import numpy as np
from model import FactoryCLIP, contrastive_loss


def load_zarr_data(filename):
    root = zarr.open(f"data/{filename}", mode="r")
    v_cmd = torch.tensor(np.array(root["velocity_cmd"]), dtype=torch.float32)
    i_cur = torch.tensor(np.array(root["motor_current"]), dtype=torch.float32)
    return v_cmd, i_cur


def train():
    # example hyperparameters
    latent_dim = 64
    batch_size = 32
    epochs = 20
    lr = 1e-3

    # load training data (Healthy tools only)
    v_train, i_train = load_zarr_data("train_healthy.zarr")
    train_ds = TensorDataset(v_train, i_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # initialize model & optimizer
    model = FactoryCLIP(latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting Training on 'Healthy' Factory Data...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_v, batch_i in train_loader:
            optimizer.zero_grad()

            z_cmd, z_cur = model(batch_v, batch_i)

            # normalize embeddings for contrastive loss
            z_cmd = F.normalize(z_cmd, dim=1)
            z_cur = F.normalize(z_cur, dim=1)

            loss = contrastive_loss(z_cmd, z_cur, model.logit_scale)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), "model_checkpoint.pt")
    print("\nModel saved to model_checkpoint.pt")


if __name__ == "__main__":
    train()
