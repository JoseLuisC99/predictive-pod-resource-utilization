import os
import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric_temporal.signal import temporal_signal_split
import lightning.pytorch as pl
from lightning import LightningModule
from lightning.pytorch.callbacks import EarlyStopping
from torch_geometric.loader import DataLoader
from torch import nn


class KubernetesA3TGCN(LightningModule):
    def __init__(self, dim_in=12, hidde_channels=10, periods=1, lr=0.1):
        super().__init__()
        self.learning_rate = lr
        self.tgnn = A3TGCN(in_channels=dim_in, out_channels=hidde_channels, periods=periods)
        self.linear = torch.nn.Linear(hidde_channels, periods)

    def __forward(self, batch):
        x = batch.x.unsqueeze(2)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        h = self.tgnn(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.linear(h)

        return h.ravel()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.__forward(batch)

    def training_step(self, batch, batch_idx):
        h = self.__forward(batch)
        y = batch.y
        loss = nn.functional.mse_loss(h, y)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def get_args():
    parser = ArgumentParser(description=("Attention Temporal Graph Convolutional Network for Traffic Forecasting model "
                                         "for Kubernetes dataset"))
    parser.add_argument("--data", type=str, required=True, default="../data",
                        help="path to dataset")
    parser.add_argument("--resource-id", type=int, default=0,
                        help="id of the pod resource")
    parser.add_argument("--lags", type=int, default=12,
                        help="number of lags for sequence")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="hidden dimension of A3TGCN model")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument("--max-epochs", type=int, default=500,
                        help="max number of epochs to train")
    parser.add_argument("--learning-rate", type=float, default=0.01,
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

    train_dataset, val_dataset = temporal_signal_split(k8s_dataset, train_ratio=0.8)
    model = KubernetesA3TGCN(args.lags, args.hidden_dim, 1, args.learning_rate)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="cuda" if args.cuda else "cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
    )
    print(len(list(train_dataset)))

    trainer.fit(
        model,
        train_dataloaders=DataLoader(list(train_dataset), batch_size=1, num_workers=23),
        val_dataloaders=DataLoader(list(val_dataset), batch_size=1, num_workers=23),
    )

    if args.output is not None:
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model = KubernetesA3TGCN.load_from_checkpoint(best_model_path)
        torch.save(best_model.state_dict(),
                   os.path.join(args.output, "a3tgcn.pt"))
