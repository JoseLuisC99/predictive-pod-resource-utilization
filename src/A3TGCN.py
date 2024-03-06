import os
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric_temporal.signal import temporal_signal_split


class KubernetesA3TGCN(torch.nn.Module):
    def __init__(self, dim_in, hidde_channels, periods):
        super().__init__()
        self.tgnn = A3TGCN(in_channels=dim_in, out_channels=hidde_channels, periods=periods)
        self.linear = torch.nn.Linear(hidde_channels, periods)

    def forward(self, x, edge_index, edge_attr):
        h = self.tgnn(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.linear(h)
        return h


def get_args():
    parser = ArgumentParser(description=("Attention Temporal Graph Convolutional Network for Traffic Forecasting model "
                                         "for Kubernetes dataset"))
    parser.add_argument("--data", type=str, required=True, default="../data",
                        help="path to dataset")
    parser.add_argument("--resource_id", type=int, default=0,
                        help="id of the pod resource")
    parser.add_argument("--lags", type=int, default=12,
                        help="number of lags for sequence")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="hidden dimension of A3TGCN model")
    parser.add_argument("--cuda", type=bool, default=False,
                        help="whether to use CUDA")
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of epochs for A3TGCN model")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--output", type=str, default=None,
                        help="path to save the model")

    return parser.parse_args()


def create_dataset(node_feats, adjacency_mat, labels, resource):
    edge_indices = []
    edge_weights = []
    features = []
    targets = []

    for i in range(len(node_feats)):
        indices, weights = dense_to_sparse(torch.from_numpy(adjacency_mat[i].mean(axis=0)))
        feature = node_feats[i, ..., resource].swapaxes(0, 1)
        target = labels[i, ..., resource]

        edge_indices.append(indices.numpy())
        edge_weights.append(weights.numpy())
        features.append(feature)
        targets.append(target)

    return DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)


if __name__ == "__main__":
    args = get_args()
    node_features = np.load(os.path.join(args.data, "node_features.npz"))
    edge_features = np.load(os.path.join(args.data, "edge_features.npz"))

    X, y = node_features["X"], node_features["y"]
    A = edge_features["A"]
    k8s_dataset = create_dataset(X, A, y, args.resource_id)

    train_dataset, test_dataset = temporal_signal_split(k8s_dataset, train_ratio=0.8)
    model = KubernetesA3TGCN(args.lags, args.hidden_dim, 1).to("cpu")

    if args.cuda:
        model.to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    model.train()
    for epoch in range(args.epochs):
        loss = 0
        step = 0
        for i, snapshot in enumerate(train_dataset):
            y_pred = model(snapshot.x.unsqueeze(2), snapshot.edge_index, snapshot.edge_attr)
            loss += torch.mean((y_pred - snapshot.y) ** 2)
            step += 1

        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        history.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:>2} | Train MSE: {loss:.4f}")

    plt.plot(history)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.show()
    if args.output is not None:
        torch.save(model.state_dict(), os.path.join(args.output, "A3TGCN.pt"))
