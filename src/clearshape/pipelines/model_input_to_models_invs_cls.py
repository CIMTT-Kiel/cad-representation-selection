# Third party imports
import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader
import pandas as pd
import warnings
import logging

# Custom imports
from clearshape.invariants.ml.modules.invs_classificator import InvariantClassifier
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


def get_dataloaders(batch_size):
    """
    DataLoader für Invariants Classification
    """
    train_loader = DataLoader(
        FabwaveDataset(
            csv_file= PATHS.DATA_MODEL_INPUT / "train.csv", 
            classification=True, 
            data_type="invariants"
        ), 
        batch_size=batch_size,
        shuffle=True
    )
    
    validation_loader = DataLoader(
        FabwaveDataset(
            csv_file=PATHS.DATA_MODEL_INPUT / "validation.csv", 
            classification=True, 
            data_type="invariants"
        ), 
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, validation_loader


def objective(trial):
    """Optuna Objective für Hyperparameter-Tuning"""
    
    # MLflow Run für diesen Trial starten
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        # Hyperparameter-Sampling
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_int("hidden_size", 64, 4096)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 2048])
        
        # Anzahl der Layer
        n_layers = trial.suggest_int("n_layers", 4, 8)
        
        # Hyperparameter zu MLflow loggen
        mlflow.log_params({
            "dropout": dropout,
            "lr": lr,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "n_layers": n_layers,
            "trial_number": trial.number,
            "in_dim": 16
        })
        
        # DataLoader
        train_loader, val_loader = get_dataloaders(batch_size=batch_size)
        num_classes = get_num_classes()
        mlflow.log_param("num_classes", num_classes)

        #depricated
        fc_layers = []
        
        current_size = hidden_size // 4
        for i in range(n_layers):
            fc_layers.append(current_size)
            
            if i < n_layers // 2:
                current_size = min(current_size * 2, hidden_size)
            else:
                current_size = max(current_size // 2, hidden_size // 4)
        
        mlflow.log_param("fc_layers", str(fc_layers))
        

        model = InvariantClassifier(
            in_dim=16,
            num_classes=num_classes,
            fc_layers=fc_layers,
            dropout=dropout,
            lr=lr
        )
        
        mlf_logger = MLFlowLogger(
            experiment_name="invariants-classification",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id  
        )
        
        # Early Stopping
        early_stop_callback = EarlyStopping(
            monitor='val_f1_score',
            patience=50,
            mode='max',
            verbose=False
        )
        
        trainer = Trainer(
            max_epochs=200,
            logger=mlf_logger,
            enable_checkpointing=False,
            enable_model_summary=True,
            enable_progress_bar=True,
            log_every_n_steps=50,
            callbacks=[early_stop_callback]
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
        val_f1_score = trainer.callback_metrics.get("val_f1_score", 0.0)
        val_accuracy = trainer.callback_metrics.get("val_acc", 0.0)
        
        if torch.is_tensor(val_loss):
            val_loss = val_loss.item()
        if torch.is_tensor(val_f1_score):
            val_f1_score = val_f1_score.item()
        if torch.is_tensor(val_accuracy):
            val_accuracy = val_accuracy.item()
        
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
    mlflow.set_experiment("invariants-classification")
    
    with mlflow.start_run(run_name="hyperparameter_optimization"):
        print("Starting hyperparameter optimization for Invariants Classification...")
        
        # Optuna Study
        study = optuna.create_study(direction="maximize")  # Maximiere F1-Score
        study.optimize(objective, n_trials=100)  
        
        best_params = study.best_trial.params
        best_value = study.best_trial.value
        
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_f1_score", best_value)
        
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best F1-Score: {best_value:.4f}")
        print(f"Best parameters: {best_params}")
    
    print("\nTraining final model with best parameters...")
    
    with mlflow.start_run(run_name="final_best_model"):
        
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "final_best_model")
        mlflow.log_param("in_dim", 16)
        
        train_loader, val_loader = get_dataloaders(
            batch_size=best_params["batch_size"]
        )
        
        num_classes = get_num_classes()
        mlflow.log_param("num_classes", num_classes)
        
        # Layer-Konfiguration rekonstruieren
        hidden_size = best_params["hidden_size"]
        n_layers = best_params["n_layers"]
        
        fc_layers = []
        current_size = hidden_size // 4
        for i in range(n_layers):
            fc_layers.append(current_size)
            if i < n_layers // 2:
                current_size = min(current_size * 2, hidden_size)
            else:
                current_size = max(current_size // 2, hidden_size // 4)
        
        mlflow.log_param("fc_layers", str(fc_layers))
        
        model = InvariantClassifier(
            in_dim=16,
            num_classes=num_classes,
            fc_layers=fc_layers,
            dropout=best_params["dropout"],
            lr=best_params["lr"]
        )
        
        mlf_logger = MLFlowLogger(
            experiment_name="invariants-classification",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_f1_score',
            patience=100,  
            mode='max',
            verbose=False
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            dirpath=PATHS.DATA_MODELS.as_posix(),
            filename='invariants-classifier',
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
            log_every_n_steps=10
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
    """
    
    mlflow.set_tracking_uri(PATHS.MLFLOW_TRACKING_URI.as_posix())
    mlflow.set_experiment("invariants-classification")
    
    with mlflow.start_run(run_name="manual_params_training"):
        
        # Manuelle Parameter
        best_params = {
            "lr": 5e-3,
            "hidden_size": 1024,
            "dropout": 0.2,
            "batch_size": 512,
            "n_layers": 6
        }
        
        mlflow.log_params(best_params)
        mlflow.log_param("training_type", "manual_params")
        mlflow.log_param("in_dim", 16)
        
        num_classes = get_num_classes()
        mlflow.log_param("num_classes", num_classes)
        
        # Layer-Konfiguration
        hidden_size = best_params["hidden_size"]
        n_layers = best_params["n_layers"]
        
        fc_layers = []
        current_size = hidden_size // 4
        for i in range(n_layers):
            fc_layers.append(current_size)
            if i < n_layers // 2:
                current_size = min(current_size * 2, hidden_size)
            else:
                current_size = max(current_size // 2, hidden_size // 4)
        
        mlflow.log_param("fc_layers", str(fc_layers))
        
        model = InvariantClassifier(
            in_dim=16,
            num_classes=num_classes,
            fc_layers=fc_layers,
            dropout=best_params["dropout"],
            lr=best_params["lr"]
        )
        
        train_loader, val_loader = get_dataloaders(best_params["batch_size"])
        
        mlf_logger = MLFlowLogger(
            experiment_name="invariants-classification",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_f1_score',
            patience=100,
            mode='max',
            verbose=False
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            dirpath=PATHS.DATA_MODELS.as_posix(),
            filename='invariants-classifier-manual',
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
            log_every_n_steps=10
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
            model_path = PATHS.DATA_MODELS / "invariants-classifier.ckpt"
        
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("evaluation_type", "test_set")
        
        model = InvariantClassifier.load_from_checkpoint(model_path)
        
        # Test Dataset
        test_dataset = FabwaveDataset(
            csv_file="/clear-shape/data/5_model_input/test.csv", 
            classification=True,
            data_type="invariants"
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=512,
            shuffle=False
        )
        
        # MLflow Logger für Evaluation
        mlf_logger = MLFlowLogger(
            experiment_name="invariants-classification",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id
        )
        
        # Evaluation
        trainer = Trainer(
            logger=mlf_logger,
            enable_progress_bar=True
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