# Third party imports
import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import joblib
from pathlib import Path
import numpy as np
import warnings
import logging

# Custom imports
from clearshape.invariants.ml.modules.invs_regressor import InvariantRegressor 
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from clearshape.dataset import FabwaveDataset
from clearshape.constants import PATHS
from clearshape.scaler.custom_scalers import LogScaler  

# Logging konfigurieren für saubere Ausgabe
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def get_dataloaders_with_scaler(batch_size):
    """
    DataLoader mit Log-Transformation für Regression
    """
    # Datasets laden
    train_dataset = FabwaveDataset(
        csv_file= PATHS.DATA_MODEL_INPUT / "train.csv", 
        classification=False,  
        regression=True,
        data_type="invariants"
    )
    
    val_dataset = FabwaveDataset(
        csv_file= PATHS.DATA_MODEL_INPUT / "validation.csv", 
        classification=False, 
        regression=True,
        data_type="invariants"
    )
    
    # Log Scaler auf Trainingsdaten fitten
    # Annahme: Dataset gibt (X, y) zurück, wobei y die 4 Targets sind
    train_targets = []
    for i in range(len(train_dataset)):
        _, targets, _ = train_dataset[i]
        train_targets.append(targets.numpy())
    
    train_targets = np.array(train_targets)
    
    # Log Scaler erstellen und fitten
    log_scaler = LogScaler(epsilon=1e-8)
    log_scaler.fit(train_targets)
    
    # Transformierte Datasets erstellen
    train_loader = DataLoader(
        LogTransformDataset(train_dataset, log_scaler, fit_scaler=False), 
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        LogTransformDataset(val_dataset, log_scaler, fit_scaler=False), 
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, log_scaler


class LogTransformDataset(torch.utils.data.Dataset):
    """
    Wrapper Dataset für Log-Transformation
    """
    def __init__(self, base_dataset, log_scaler, fit_scaler=False):
        self.base_dataset = base_dataset
        self.log_scaler = log_scaler
        
        # Alle Targets sammeln und transformieren
        self.transformed_data = []
        
        for i in range(len(base_dataset)):
            x, y, metadata = base_dataset[i]
            
            # Y transformieren
            y_np = y.numpy().reshape(1, -1)  # (1, n_targets) für scaler
            if fit_scaler and i == 0:
                y_transformed = self.log_scaler.fit_transform(y_np)
            else:
                y_transformed = self.log_scaler.transform(y_np)
            
            y_transformed = torch.FloatTensor(y_transformed.flatten())
            
            self.transformed_data.append((x, y_transformed, metadata))
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.transformed_data[idx]


def objective(trial):
    """Optuna Objective für Hyperparameter-Tuning"""
    
    # MLflow Run für diesen Trial starten
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        # Hyperparameter-Sampling für Regression
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Architektur-Parameter
        hidden_size = trial.suggest_int("hidden_size", 256, 2048)
        n_layers = trial.suggest_int("n_layers", 3, 8)
        use_target_heads = trial.suggest_categorical("use_target_heads", [False])
        
        batch_size = trial.suggest_categorical("batch_size", [128,256,2048])
        
        # Hyperparameter zu MLflow loggen
        mlflow.log_params({
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "use_target_heads": use_target_heads,
            "batch_size": batch_size,
            "trial_number": trial.number
        })
        
        # DataLoader mit Log-Transformation
        train_loader, val_loader, log_scaler = get_dataloaders_with_scaler(batch_size=batch_size)
        
        # Dynamische Layer-Konfiguration
        fc_layers = []
        current_size = hidden_size // 4
        for i in range(n_layers):
            fc_layers.append(current_size)
            if i < n_layers // 2:  # Erste Hälfte: vergrößern
                current_size = min(current_size * 2, hidden_size)
            else:  # Zweite Hälfte: verkleinern
                current_size = max(current_size // 2, hidden_size // 8)
        
        # Regressionsmodell erstellen
        model = InvariantRegressor(
            in_dim=16,
            n_targets=4,  # VOLUME, FACES, EDGES, VERTICES
            fc_layers=fc_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            use_target_specific_heads=use_target_heads,
            log_scaler=log_scaler,
            target_names=['VOLUME', 'FACES', 'EDGES', 'VERTICES']
        )
        
        # MLflow Logger für Lightning
        mlf_logger = MLFlowLogger(
            experiment_name="invariants-regression",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id  
        )
        
        # Early Stopping für Regression
        early_stop_callback = EarlyStopping(
            monitor='val_mse_original',  
            patience=100,
            mode='min',
            verbose=False  
        )
        
        trainer = Trainer(
            max_epochs=1000,
            logger=mlf_logger,
            enable_checkpointing=True,
            enable_model_summary=True,
            enable_progress_bar=True,
            log_every_n_steps=50,  
            callbacks=[early_stop_callback],
            accelerator="cuda",
            devices=1
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        # get metrics
        val_mse = trainer.callback_metrics.get("val_mse_original", float('inf')).item()
        val_mae = trainer.callback_metrics.get("val_mae_original", float('inf')).item()
        val_loss = trainer.callback_metrics.get("val_loss", float('inf')).item()
        
        mlflow.log_metrics({
            "val_mse_original": val_mse,
            "val_mae_original": val_mae,
            "val_loss": val_loss,
            "final_epoch": trainer.current_epoch
        })
        
        print(f"Trial {trial.number}: MSE={val_mse:.6f}, MAE={val_mae:.6f}")
        
        return val_mse


def main():
    """Hauptfunktion für Hyperparameter-Tuning und finales Training"""
    print("Starting main function for hyperparameter tuning and final training...")
    # MLflow Setup
    mlflow.set_tracking_uri(PATHS.MLFLOW_TRACKING_URI.as_posix())
    mlflow.set_experiment("invariants-regression")

    # Hauptrun für das gesamte Experiment starten
    #with mlflow.start_run(run_name="hyperparameter_optimization"):
    #    print("Starting hyperparameter optimization...")
    #    
    #    # Optuna Study
    #    study = optuna.create_study(direction="minimize")  # Minimiere MSE
    #    study.optimize(objective, n_trials=200)  
    #    
    #    # Beste Parameter loggen
    #    best_params = study.best_trial.params
    #    best_value = study.best_trial.value
    #    
    #    mlflow.log_params(best_params)
    #    mlflow.log_metric("best_val_mse", best_value)
    #    
    #    print(f"\nBest trial: {study.best_trial.number}")
    #    print(f"Best MSE: {best_value:.6f}")
    #    print(f"Best parameters: {best_params}")
    

    print("\nTraining final model with best parameters...")
    
    with mlflow.start_run(run_name="final_best_model"):

        best_params={
            'dropout' : 0.18243074649601815,
            'lr' : 0.0026224079062513784,
            'weight_decay' : 1.050686152787924e-05,
            'hidden_size' : 1676,
            'n_layers' : 8,
            'use_target_heads' : False,
            'batch_size' : 2048,
            'in_dim' : 16,
            'fc_layers' : [419, 838, 1676, 1676, 1676, 838, 419, 209],
            'use_target_specific_heads' : False,
        }

        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "final_best_model")

        # get dataloaders and scaler
        train_loader, val_loader, log_scaler = get_dataloaders_with_scaler(
            batch_size=best_params["batch_size"]
        )
        
        # init model with best params
        hidden_size = best_params["hidden_size"]
        n_layers = best_params["n_layers"]
        
        fc_layers = []
        current_size = hidden_size // 4
        for i in range(n_layers):
            fc_layers.append(current_size)
            if i < n_layers // 2:
                current_size = min(current_size * 2, hidden_size)
            else:
                current_size = max(current_size // 2, hidden_size // 8)
        
        # # init model with best params
        model = InvariantRegressor(
            in_dim=16,
            n_targets=4,
            fc_layers=fc_layers,
            dropout=best_params["dropout"],
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            use_target_specific_heads=best_params["use_target_heads"],
            log_scaler=log_scaler,
            target_names=['VOLUME', 'FACES', 'EDGES', 'VERTICES']
        )
        
        # MLflow Logger für finales Training
        mlf_logger = MLFlowLogger(
            experiment_name="invariants-regression",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id
        )
        
        # Callbacks für finales Training
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=100,  
            mode='min',
            verbose=False
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_mse_original',
            save_top_k=3,
            mode='min',
            dirpath=PATHS.DATA_MODELS.as_posix(),
            filename='invariants-regressor',
            save_weights_only=False,
            verbose=False
        )
        
        # Finaler Trainer
        trainer = Trainer(
            max_epochs=1000,  # Mehr Epochs für finales Training
            logger=mlf_logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            enable_checkpointing=True,
            enable_progress_bar=True,  # Für finales Training anzeigen
            log_every_n_steps=10,
            accelerator="cuda",
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
        
        # Log Scaler separat speichern und zu MLflow hinzufügen
        scaler_path = PATHS.DATA_MODELS / "log_scaler.joblib"
        joblib.dump(log_scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path), "scaler")
        
        print("Final training completed!")


def evaluate_model(model_path=None, scaler_path=None):
    """Separate Funktion für Modell-Evaluation auf Test-Set"""
    
    with mlflow.start_run(run_name="model_evaluation"):
        # Pfade anpassen
        if model_path is None:
            model_path = "path/to/best/model.ckpt"  # Aus Training
        if scaler_path is None:
            scaler_path = PATHS.DATA_MODELS / "log_scaler.joblib"
        
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("scaler_path", str(scaler_path))
        
        log_scaler = joblib.load(scaler_path)
        model = InvariantRegressor.load_from_checkpoint(model_path, log_scaler=log_scaler)
        
        # Test Dataset
        test_dataset = FabwaveDataset(
            csv_file=PATHS.DATA_MODEL_INPUT / "test.csv", 
            classification=False,
            regression=True,
            data_type="invariants"
        )
        
        test_loader = DataLoader(
            LogTransformDataset(test_dataset, log_scaler, fit_scaler=False),
            batch_size=512,
            shuffle=False
        )
        
        # MLflow Logger für Evaluation
        mlf_logger = MLFlowLogger(
            experiment_name="invariants-regression",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id
        )
        
        # Evaluation
        trainer = Trainer(
            logger=mlf_logger,
            enable_progress_bar=False
        )
        
        results = trainer.test(model, test_loader)
        
        # Test-Ergebnisse loggen
        if results:
            for key, value in results[0].items():
                mlflow.log_metric(f"test_{key}", value)
        
        print("Model evaluation completed!")
        return results


if __name__ == "__main__":
    main()
    
    # Optional: Test-Evaluation
    # evaluate_model()