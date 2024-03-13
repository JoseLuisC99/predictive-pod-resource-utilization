import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric_temporal.signal import temporal_signal_split
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, RecurrentNetwork
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
import torchmetrics
import optuna
from argparse import ArgumentParser
from functools import partial


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


def train(model, train_dataset, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(150):
        loss = 0
        step = 0
        for i, snapshot in enumerate(train_dataset):
            y_pred = model(snapshot.x.unsqueeze(2), snapshot.edge_index, snapshot.edge_attr)
            loss += torch.mean((y_pred - snapshot.y) ** 2)
            step += 1
        loss /= (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, test_dataset):
    model.eval()
    loss = 0
    step = 0
    for i, snapshot in enumerate(test_dataset):
        y_pred = model(snapshot.x.unsqueeze(2), snapshot.edge_index, snapshot.edge_attr)
        loss += torch.mean((y_pred - snapshot.y) ** 2)
        step += 1
    loss /= (step + 1)
    return loss


def objective_astgcn(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 4, 128)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)

    node_features = np.load("data/node_features.npz")
    edge_features = np.load("data/edge_features.npz")
    X, y = node_features["X"], node_features["y"]
    A = edge_features["A"]
    k8s_dataset = create_dataset(X, A, y, 0)
    train_dataset, test_dataset = temporal_signal_split(k8s_dataset, train_ratio=0.8)

    model = KubernetesA3TGCN(12, hidden_dim, 1).to("cpu")
    train(model, train_dataset, learning_rate)
    return test(model, test_dataset)


def get_data_from_csv():
    data = pd.read_csv("data/pod_metrics.csv")
    data["group"] = 0
    data["time_idx"] = data["timestamp"].argsort()

    drop_names = []
    for name in data.columns.values:
        if name.endswith("mem"):
            drop_names.append(name)
    data = data.drop(columns=drop_names)
    targets = list(data.columns.values)
    targets.remove("timestamp")
    targets.remove("group")
    targets.remove("time_idx")

    training = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target=targets,
        group_ids=["group"],
        max_encoder_length=12,
        max_prediction_length=1,
        static_reals=["group"],
        time_varying_known_reals=["timestamp", "time_idx"],
        time_varying_unknown_reals=targets
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=128)
    val_dataloader = validation.to_dataloader(train=False, batch_size=128)

    return training, train_dataloader, val_dataloader


def get_tft(trial, training):
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 5)
    hidden_size = trial.suggest_int("hidden_size", 8, 512)
    attention_head_size = trial.suggest_int("attention_head_size", 1, 10)
    dropout = trial.suggest_float("dropout", 0.1, 0.8)
    hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 4, 32)

    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        lstm_layers=lstm_layers,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )


def get_lstm(trial, training):
    hidden_size = trial.suggest_int("hidden_size", 8, 512)
    dropout = trial.suggest_float("dropout", 0.1, 0.8)
    rnn_layers = trial.suggest_int("lstm_layers", 1, 10)
    cell_type = trial.suggest_categorical("cell_type", ["LSTM", "GRU"])

    return RecurrentNetwork.from_dataset(
        training,
        dropout=dropout,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        cell_type=cell_type
    )


def objective_recurrent(trial, architecture):
    training, train_dataloader, val_dataloader = get_data_from_csv()
    if architecture == "tft":
        model = get_tft(trial, training)
    elif architecture == "rnn":
        model = get_lstm(trial, training)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 100.0)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cuda",
        enable_model_summary=True,
        gradient_clip_val=gradient_clip_val,
        limit_train_batches=50,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    mse = torchmetrics.regression.MeanSquaredError().to("cuda")
    predictions = model.predict(val_dataloader, return_y=True)
    return mse(torch.cat(predictions.output), torch.cat(predictions.y[0]))


if __name__ == "__main__":
    parser = ArgumentParser("Hyperparameter Search")
    parser.add_argument("--model", type=str, help="Model to tune hyperparameters")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--resource", type=int, default=0, help="Resource to predict")
    parser.add_argument("--study-name", type=str, help="Experiment's name for Optuna")
    args = parser.parse_args()

    try:
        study = optuna.create_study(
            storage="sqlite:///pycache/hypertunning.db",  # Specify the storage URL here.
            study_name=args.study_name
        )
    except optuna.exceptions.DuplicatedStudyError:
        # print(type(e))
        study = optuna.load_study(
            storage="sqlite:///pycache/hypertunning.db",  # Specify the storage URL here.
            study_name=args.study_name
        )

    if args.model == "gcn":
        objective = objective_astgcn
    elif args.model == "tft" or args.model == "rnn":
        objective = partial(objective_recurrent, architecture=args.model)
    else:
        raise ValueError(f"Objective {args.model} not valid")

    study.optimize(objective, n_trials=args.trials)
    print(study.best_params)

