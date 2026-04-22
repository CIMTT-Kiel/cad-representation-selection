# Third party imports
import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader
import joblib
from pathlib import Path
import warnings
import logging
import pandas as pd

# Enable Flash Attention via PyTorch SDPA backend (requires CUDA + PyTorch >= 2.0)
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)

# Custom imports
from clearshape.vecsets.ml.modules.trsfm_classificator import VecsetClassifierModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from clearshape.dataset import FabwaveDataset
from clearshape.constants import PATHS

# Logging konfigurieren für saubere Ausgabe
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def get_num_classes():
    """Dynamisch die Anzahl der Klassen aus dem Target-File lesen"""
    return len(pd.read_csv(PATHS.DATA_FEATURE / "fabwave_targets.csv")["class_id"].unique())

def get_dataloaders(batch_size=32):
    """
    DataLoader für Vecsets Classification
    """
    train_loader = DataLoader(
        FabwaveDataset(
            csv_file=PATHS.DATA_MODEL_INPUT/ "train.csv", 
            classification=True, 
            data_type="vecsets"
        ), 
        batch_size=batch_size,
        shuffle=True
    )
    
    validation_loader = DataLoader(
        FabwaveDataset(
            csv_file=PATHS.DATA_MODEL_INPUT / "validation.csv", 
            classification=True, 
            data_type="vecsets"
        ), 
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, validation_loader


def objective(trial):
    """Optuna Objective für Hyperparameter-Tuning mit Constraint-Checking"""

    num_classes = get_num_classes()


    # MLflow Run für diesen Trial starten
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        # Hyperparameter-Sampling
        dropout = trial.suggest_float("dropout", 0.3, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256])

        # D_model must be divisible by nhead
        d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512, 768, 1024])

        # Transformer-specific hyperparameters
        nhead = trial.suggest_categorical("nhead", [2, 4, 8, 16])
        num_layers = trial.suggest_int("num_layers", 2, 8)

        # Feedforward network size
        dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 512, 1024])
        
        # CONSTRAINT: d_model muss durch nhead teilbar sein
        if d_model % nhead != 0:
            mlflow.log_params({
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "d_model": d_model,
                "trial_number": trial.number,
                "pruned": True,
                "prune_reason": f"d_model ({d_model}) not divisible by nhead ({nhead})"
            })
            print(f"Trial {trial.number} pruned: d_model={d_model} not divisible by nhead={nhead}")
            raise optuna.exceptions.TrialPruned()
        
        # Hyperparameter zu MLflow loggen
        mlflow.log_params({
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "d_model": d_model,
            "trial_number": trial.number,
            "input_dim": 32,
            "num_classes": num_classes,
            "pruned": False
        })
        
        # DataLoader
        train_loader, val_loader = get_dataloaders(batch_size=batch_size)
        
        # Model erstellen
        model = VecsetClassifierModule( 
            lr=lr,
            weight_decay=weight_decay,
            input_dim=32, 
            d_model=d_model,  # Verwende die Variable statt hartkodiert 1024
            nhead=nhead, 
            num_layers=num_layers, 
            num_classes=num_classes, 
            dim_feedforward=dim_feedforward, 
            fc_layers=None,
            dropout=dropout
        )
        
        # MLflow Logger für Lightning
        mlf_logger = MLFlowLogger(
            experiment_name="vecsets-classification",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id  
        )
        
        # Early Stopping
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            verbose=False
        )
        
        trainer = Trainer(
            max_epochs=50,  # Genug Epochs für Trials
            logger=mlf_logger,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True,
            log_every_n_steps=10,
            callbacks=[early_stop_callback],
            precision="bf16-mixed" if torch.cuda.is_available() else '32',  # Mixed precision
            devices=1
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        # Metriken extrahieren - robust gegen Tensor und Float
        val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
        val_f1_score = trainer.callback_metrics.get("val_f1_score", 0.0)
        val_accuracy = trainer.callback_metrics.get("val_acc", 0.0)
        
        # Konvertiere zu Python float falls Tensor
        if torch.is_tensor(val_loss):
            val_loss = val_loss.item()
        if torch.is_tensor(val_f1_score):
            val_f1_score = val_f1_score.item()
        if torch.is_tensor(val_accuracy):
            val_accuracy = val_accuracy.item()
        
        # Metriken zu MLflow loggen
        mlflow.log_metrics({
            "val_loss": val_loss,
            "val_f1_score": val_f1_score,
            "val_acc": val_accuracy,
            "final_epoch": trainer.current_epoch
        })
        
        print(f"Trial {trial.number}: F1={val_f1_score:.4f}, Loss={val_loss:.4f}, Acc={val_accuracy:.4f}")
        
        return val_f1_score


def main():
    """Hauptfunktion für Hyperparameter-Tuning und finales Training"""

    # MLflow Setup
    mlflow.set_tracking_uri(PATHS.MLFLOW_TRACKING_URI.as_posix())
    mlflow.set_experiment("vecsets-classification")

    # Hauptrun für das gesamte Experiment starten
    with mlflow.start_run(run_name="hyperparameter_optimization"):
        print("Starting hyperparameter optimization for Vecsets Classification...")

        # Optuna Study mit Pruning für bessere Effizienz
        study = optuna.create_study(
            direction="maximize",  # Maximiere F1-Score
            sampler=optuna.samplers.TPESampler(n_startup_trials=10),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        study.optimize(objective, n_trials=100)  # 30 Trials für gute Exploration

        # Beste Parameter loggen
        best_params = study.best_trial.params
        best_value = study.best_trial.value

        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_f1_score", best_value)
        mlflow.log_metric("n_trials", len(study.trials))
        mlflow.log_metric("n_completed_trials", len([t for t in study.trials if t.state.name == 'COMPLETE']))

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best F1-Score: {best_value:.4f}")
        print(f"Best parameters: {best_params}")

        # Save Optuna study
        study_path = PATHS.DATA_MODELS / "vecsets_classifier_optuna_study.joblib"
        joblib.dump(study, study_path)
        mlflow.log_artifact(str(study_path), "optuna_study")

    print("\nTraining final model with best parameters...")
    
    # Finales Training mit besten Parametern
    with mlflow.start_run(run_name="final_best_model"):
        num_classes = get_num_classes()
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "final_best_model")
        mlflow.log_param("input_dim", 32)
        mlflow.log_param("num_classes", num_classes)
        
        # DataLoader mit besten Parametern
        train_loader, val_loader = get_dataloaders(
            batch_size=best_params.get("batch_size", 64)
        )
        

        # Finales Model mit besten Parametern
        model = VecsetClassifierModule( 
            lr=best_params["lr"],
            weight_decay=best_params.get("weight_decay", 1e-4),  # Fallback für alte Runs
            input_dim=32, 
            d_model=best_params.get("d_model", 1024), 
            nhead=best_params["nhead"], 
            num_layers=best_params["num_layers"], 
            num_classes=num_classes, 
            dim_feedforward=best_params["dim_feedforward"], 
            fc_layers=None,
            dropout=best_params["dropout"], 
        )
        
        # MLflow Logger für finales Training
        mlf_logger = MLFlowLogger(
            experiment_name="vecsets-classification",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id
        )
        
        # Callbacks für finales Training
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=20,  # Mehr Geduld für finales Training
            mode='min',
            verbose=False
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1_score',
            save_top_k=1,
            mode='max',
            dirpath=PATHS.DATA_MODELS.as_posix(),
            filename='vecsets-classifier',
            save_weights_only=False,
            verbose=False
        )
        
        # Finaler Trainer
        trainer = Trainer(
            max_epochs=1000,
            logger=mlf_logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            enable_checkpointing=True,
            enable_model_summary=False,
            enable_progress_bar=True,
            log_every_n_steps=10,
            precision="bf16-mixed" if torch.cuda.is_available() else '32',  # Mixed precision
            devices=1
        )
        
        # Training
        trainer.fit(model, train_loader, val_loader)
        
        # Finale Metriken loggen
        final_metrics = trainer.callback_metrics
        for key, value in final_metrics.items():
            if torch.is_tensor(value):
                mlflow.log_metric(f"final_{key}", value.item())
        
        # Modell-Artefakte zu MLflow hinzufügen
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            mlflow.log_artifact(best_model_path, "model")
            print(f"Best model saved to: {best_model_path}")
        
        print("Final training completed!")


def train_with_manual_params():
    """
    Optionale Funktion: Training mit manuell gesetzten Parametern
    (z.B. wenn Sie bereits optimale Parameter aus früheren Runs kennen)
    """
    
    mlflow.set_tracking_uri(PATHS.MLFLOW_TRACKING_URI.as_posix())
    mlflow.set_experiment("vecsets-classification")
    
    with mlflow.start_run(run_name="manual_params_training"):
        
        num_classes = get_num_classes()

        # Manuelle Parameter (aus früheren Runs)
        best_params = {
            "lr": 2.53e-4,
            "weight_decay": 1e-4,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 1211,
            "dropout": 0.42,
            "batch_size": 64,
            "d_model": 1024
        }
        
        mlflow.log_params(best_params)
        mlflow.log_param("training_type", "manual_params")
        mlflow.log_param("d_model", 1024)
        mlflow.log_param("input_dim", 32)
        mlflow.log_param("num_classes", num_classes)
        
        model = VecsetClassifierModule( 
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            input_dim=32, 
            d_model=best_params["d_model"], 
            nhead=best_params["nhead"], 
            num_layers=best_params["num_layers"], 
            num_classes=num_classes, 
            dim_feedforward=best_params["dim_feedforward"], 
            fc_layers=None,
            dropout=best_params["dropout"], 
        )
        
        train_loader, val_loader = get_dataloaders(best_params["batch_size"])
        
        mlf_logger = MLFlowLogger(
            experiment_name="vecsets-classification",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=200,
            mode='min',
            verbose=False
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1_score',
            save_top_k=1,
            mode='max',
            dirpath=PATHS.DATA_MODELS.as_posix(),
            filename='vecsets-classifier-manual',
            save_weights_only=False,
            verbose=False
        )
        
        trainer = Trainer(
            max_epochs=1000, 
            logger=mlf_logger, 
            callbacks=[early_stop_callback, checkpoint_callback], 
            enable_checkpointing=True, 
            enable_model_summary=False, 
            enable_progress_bar=True,
            log_every_n_steps=10,
            precision="bf16-mixed" if torch.cuda.is_available() else '32',  # Mixed precision
            devices=1
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        # Finale Metriken loggen
        final_metrics = trainer.callback_metrics
        for key, value in final_metrics.items():
            if torch.is_tensor(value):
                mlflow.log_metric(f"final_{key}", value.item())
        
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            mlflow.log_artifact(best_model_path, "model")
            print(f"Model saved to: {best_model_path}")


def evaluate_model(model_path=None):
    """Separate Funktion für Modell-Evaluation auf Test-Set"""
    
    with mlflow.start_run(run_name="model_evaluation"):
        
        if model_path is None:
            model_path = PATHS.DATA_MODELS / "vecsets-classifier.ckpt"
        
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("evaluation_type", "test_set")
        
        model = VecsetClassifierModule.load_from_checkpoint(model_path)
        
        # Test Dataset
        test_dataset = FabwaveDataset(
            csv_file="/clear-shape/data/5_model_input/test.csv", 
            classification=True,
            data_type="vecsets"
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False
        )
        
        # MLflow Logger für Evaluation
        mlf_logger = MLFlowLogger(
            experiment_name="vecsets-classification",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id
        )
        
        # Evaluation
        trainer = Trainer(
            logger=mlf_logger,
            enable_progress_bar=True,
            precision="bf16-mixed" if torch.cuda.is_available() else '32',  # Mixed precision
            devices=1
        )
        
        results = trainer.test(model, test_loader)
        
        # Test-Ergebnisse loggen
        if results:
            for key, value in results[0].items():
                mlflow.log_metric(f"test_{key}", value)
        
        print("Model evaluation completed!")
        return results


if __name__ == "__main__":
    # Haupt-Training mit Hyperparameter-Optimierung
    main()
    
    # Optional: Training mit manuellen Parametern
    # train_with_manual_params()
    
    # Optional: Test-Evaluation
    # evaluate_model()