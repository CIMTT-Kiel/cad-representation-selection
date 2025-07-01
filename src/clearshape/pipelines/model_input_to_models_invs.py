import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

from clearshape.invariants.ml.modules.invs_classificator import InvariantClassifier


def get_dataloaders(batch_size=32):
    # Dummy-Daten (ersetzen durch echte)
    X = torch.randn(1000, 16)
    y = torch.randint(0, 38, (1000,))
    ds = TensorDataset(X, y)
    train_ds, val_ds = random_split(ds, [800, 200])
    return DataLoader(train_ds, batch_size=batch_size), DataLoader(val_ds, batch_size=batch_size)

def objective(trial):
    # Hyperparameter-Sampling

    dropout = trial.suggest_float("dropout", 0.0, 0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 64, 512)

    batch_size = trial.suggest_categorical("batch_size", [256,512,1024])


    train_loader, val_loader = get_dataloaders(batch_size)

    model = InvariantClassifier(
        in_dim=16,
        num_classes=38,
        fc_layers=[hidden_size, 2*hidden_size, hidden_size],
        dropout=dropout,
        lr=lr
    )

    # MLflow Logger
    mlf_logger = MLFlowLogger(experiment_name="invariants-classification", tracking_uri="file:./../invariants/ml/mlruns")

    trainer = Trainer(
        max_epochs=100,
        logger=mlf_logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=1
    )

    trainer.fit(model, train_loader, val_loader)
    val_acc = trainer.callback_metrics["val_acc"].item()

    # Hyperparameter und Metrik an MLflow loggen (über den Logger)
    mlf_logger.log_hyperparams(trial.params)
    mlf_logger.log_metrics({"val_acc": val_acc})

    return val_acc  # Optuna maximiert diese Metrik


def main():
    mlflow.set_tracking_uri("file:./../invariants/ml/mlruns")  # Oder HTTP URI für Server
    mlflow.set_experiment("invariants-classification")

    mlf_logger = MLFlowLogger(experiment_name="invariants-classification", tracking_uri="file:./../invariants/ml/mlruns", run_name="best-model")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Beste Konfiguration:", study.best_trial.params)

    # Bestes Modell trainieren und speichern
    best_params = study.best_trial.params

    hidden_size = best_params["hidden_size"]
    model = InvariantClassifier(
        in_dim=16,
        num_classes=38,
        fc_layers=[hidden_size, 2*hidden_size, hidden_size],
        dropout=best_params["dropout"],
        lr=best_params["lr"]
    )

    train_loader, val_loader = get_dataloaders(best_params["batch_size"])


    trainer = Trainer(max_epochs=100, logger=mlf_logger)
    trainer.fit(model, train_loader, val_loader)


    torch.save(model.state_dict(), r"../models/invariants-classification.pt")


if __name__ == "__main__":
    main()