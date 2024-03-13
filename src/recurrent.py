import os
import shutil
import warnings
import torch
from argparse import ArgumentParser
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, RecurrentNetwork
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')


def get_args():
    parser = ArgumentParser(description="Recurrent models for Kubernetes dataset")
    parser.add_argument("--data", type=str, required=True,
                        help="path to dataset")
    parser.add_argument("--model", type=str, required=True, help="Model to use [tft or recurrent]")
    parser.add_argument("--resource-id", type=int, default=0,
                        help="id of the pod resource")
    parser.add_argument("--lags", type=int, default=12,
                        help="number of lags for sequence")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="batch size for episode")
    parser.add_argument("--rnn-layers", type=int, default=2,
                        help="Number of RNN layers - important hyperparameter")
    parser.add_argument("--hidden-size", type=int, default=10,
                        help="Hidden size of network")
    parser.add_argument("--attn-head-size", type=int, default=4,
                        help="Number of attention heads (TFT only)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max-epochs", type=int, default=500,
                        help="maximum number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="learning rate for the optimizer")
    parser.add_argument("--cell-type", type=str, default="LSTM",
                        help='Recurrent cell type [“LSTM”, “GRU”]')
    parser.add_argument("--hidden-continuous-size", type=int, default=8)
    parser.add_argument("--gradient-clip-val", type=float, default=0.1)
    parser.add_argument("--output", type=str, default=None,
                        help="path to save the model")

    return parser.parse_args()


def get_dataset(data, resource_id, lags):
    data = pd.read_csv(os.path.join(data, "pod_metrics.csv"))
    data["group"] = 0
    data["time_idx"] = data["timestamp"].argsort()

    drop_names = []
    for name in data.columns.values:
        if name.endswith("mem" if resource_id == 0 else "cpu"):
            drop_names.append(name)
    data = data.drop(columns=drop_names)

    # Loading the dataset
    targets = list(data.columns.values)
    targets.remove("timestamp")
    targets.remove("group")
    targets.remove("time_idx")

    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target=targets,
        group_ids=["group"],
        max_encoder_length=lags,
        max_prediction_length=1,
        static_reals=["group"],
        time_varying_known_reals=["timestamp", "time_idx"],
        time_varying_unknown_reals=targets
    ), data


if __name__ == "__main__":
    args = get_args()
    training, data = get_dataset(args.data, args.resource_id, args.lags)

    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    train_dataloader = training.to_dataloader(train=True, batch_size=args.batch_size)
    val_dataloader = validation.to_dataloader(train=False, batch_size=args.batch_size)

    # Defining the model
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    if args.model == "tft":
        model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=args.learning_rate,
            lstm_layers=args.rnn_layers,
            hidden_size=args.hidden_size,
            attention_head_size=args.attn_head_size,
            dropout=args.dropout,
            hidden_continuous_size=args.hidden_continuous_size,
            loss=QuantileLoss(),
            log_interval=10,
            optimizer="Ranger",
            reduce_on_plateau_patience=4,
        )
    else:
        model = RecurrentNetwork.from_dataset(
            training,
            dropout=args.dropout,
            hidden_size=args.hidden_size,
            rnn_layers=args.rnn_layers,
            cell_type=args.cell_type
        )
    print(f"Number of parameters in network: {model.size() / 1e3:.1f}k")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="cuda",
        enable_model_summary=True,
        gradient_clip_val=args.gradient_clip_val,
        limit_train_batches=50,
        callbacks=[early_stop_callback],
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Saving the model
    if args.output is not None:
        best_model_path = trainer.checkpoint_callback.best_model_path
        shutil.copy2(
            best_model_path,
            os.path.join(
                args.output,
                f"{args.model}{f'_{args.cell_type}' if args.model == 'recurrent' else ''}.ckpt".lower()))

