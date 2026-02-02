# Third party imports
import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import joblib
import warnings
import logging
import sys

# Custom imports
from clearshape.vecsets.ml.modules.trsfm_regressor import VecsetTransformerRegressor
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from clearshape.dataset import FabwaveDataset
from clearshape.constants import PATHS

# Log Scaler Import
from clearshape.scaler.custom_scalers import LogScaler 

# Logging konfigurieren für saubere Ausgabe
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


def get_dataloaders_with_scaler(batch_size):
    """
    DataLoader mit Log-Transformation für Transformer Regression
    """
    # Datasets laden
    train_dataset = FabwaveDataset(
        csv_file="/clear-shape/data/5_model_input/train.csv", 
        classification=False,  
        regression=True,
        data_type="vecsets"
    )
    
    val_dataset = FabwaveDataset(
        csv_file="/clear-shape/data/5_model_input/validation.csv", 
        classification=False,  
        regression=True,
        data_type="vecsets"
    )
    
    # Log Scaler auf Trainingsdaten fitten
    train_targets = []
    for i in range(len(train_dataset)):
        _, targets, _ = train_dataset[i]
        train_targets.append(targets.numpy())
    
    train_targets = np.array(train_targets)
    
    log_scaler = LogScaler(epsilon=1e-8)
    log_scaler.fit(train_targets)
    
    train_loader = DataLoader(
        LogTransformDataset(train_dataset, log_scaler, fit_scaler=False), 
        batch_size=batch_size,
        shuffle=True,


    )
    
    val_loader = DataLoader(
        LogTransformDataset(val_dataset, log_scaler, fit_scaler=False), 
        batch_size=batch_size,
        shuffle=False,


    )
    
    return train_loader, val_loader, log_scaler


class LogTransformDataset(torch.utils.data.Dataset):
    """
    Wrapper Dataset für Log-Transformation
    """
    def __init__(self, base_dataset, log_scaler, fit_scaler=False):
        self.base_dataset = base_dataset
        self.log_scaler = log_scaler
        
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
            
            self.transformed_data.append((x, y_transformed, metadata ))
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.transformed_data[idx]


def objective(trial):
    """Optuna Objective für Transformer Hyperparameter-Tuning"""
    
    # MLflow Run für diesen Trial starten
    with mlflow.start_run(run_name=f"transformer_trial_{trial.number}", nested=True):

        embed_dim = trial.suggest_int("embed_dim", 128, 512, step=64)  
        num_heads = trial.suggest_int("num_heads", 4, 16, step=2)      
        
        # check if embed_dim is divisible by num_heads - if not, prune the trial
        if embed_dim % num_heads != 0:
            raise optuna.TrialPruned(f"embed_dim={embed_dim} not divisible by num_heads={num_heads}")
        
        num_layers = trial.suggest_int("num_layers", 2, 8)
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        
        # Training Hyperparameter
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        warmup_epochs = trial.suggest_int("warmup_epochs", 5, 20)
        
        # Architecture options
        use_target_heads = trial.suggest_categorical("use_target_heads", [True, False])
        
        batch_size = 64
        
        # Hyperparameter zu MLflow loggen
        mlflow.log_params({
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "use_target_heads": use_target_heads,
            "batch_size": batch_size,
            "trial_number": trial.number,
            "model_type": "transformer"
        })
        
        try:
            # DataLoader mit Log-Transformation
            train_loader, val_loader, log_scaler = get_dataloaders_with_scaler(batch_size=batch_size)
            
            # Transformer Modell
            model = VecsetTransformerRegressor(
                input_dim=32,
                n_targets=4,  # VOLUME, FACES, EDGES, VERTICES
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                use_target_specific_heads=use_target_heads,
                lr=lr,
                weight_decay=weight_decay,
                max_epochs=150,  
                warmup_epochs=warmup_epochs,
                log_scaler=log_scaler,
                target_names=['VOLUME', 'FACES', 'EDGES', 'VERTICES']
            )
            
            # Model parameters count
            model_params = sum(p.numel() for p in model.parameters())
            mlflow.log_metric("model_parameters", model_params)
            
            # MLflow Logger für Lightning
            mlf_logger = MLFlowLogger(
                experiment_name="vecset-regression",
                tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
                run_id=mlflow.active_run().info.run_id
            )
            
            # Early Stopping für Transformer
            early_stop_callback = EarlyStopping(
                monitor='val_mse_original',  
                patience=20,  
                mode='min',
                verbose=False
            )
            
            # Trainer 
            trainer = Trainer(
                max_epochs=150,
                logger=mlf_logger,
                enable_checkpointing=False,
                enable_model_summary=True,
                enable_progress_bar=True,  
                log_every_n_steps=50,
                callbacks=[early_stop_callback],
                gradient_clip_val=1.0,  
                precision='16-mixed' if torch.cuda.is_available() else '32',  # Mixed precision
                accumulate_grad_batches=1,
                deterministic=False,  
                accelerator="auto"
            )
            
            trainer.fit(model, train_loader, val_loader)
            
            # Metriken extrahieren
            val_mse = trainer.callback_metrics.get("val_mse_original", float('inf')).item()
            val_mae = trainer.callback_metrics.get("val_mae_original", float('inf')).item()
            val_loss = trainer.callback_metrics.get("val_loss", float('inf')).item()
            
            # Finale Metriken zu MLflow loggen
            mlflow.log_metrics({
                "val_mse_original": val_mse,
                "val_mae_original": val_mae,
                "val_loss": val_loss,
                "final_epoch": trainer.current_epoch,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_layers": num_layers
            })
            
            print(f"Trial {trial.number}: MSE={val_mse:.6f}, MAE={val_mae:.6f}, Params={model_params:,}")
            
            return val_mse
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            print(f"Trial {trial.number} failed: {e}")
            return float('inf')


def main():
    """Hauptfunktion für Transformer Hyperparameter-Tuning und finales Training"""
    
    # MLflow Setup
    mlflow.set_tracking_uri(PATHS.MLFLOW_TRACKING_URI.as_posix())
    mlflow.set_experiment("vecset-regression")
    
    # Hauptrun für das gesamte Experiment starten
    with mlflow.start_run(run_name="transformer_hyperparameter_optimization"):

        mlflow.log_param("model_architecture", "transformer")
        mlflow.log_param("optimization_algorithm", "optuna_tpe")

        # Optuna Study mit Transformer-spezifischen Einstellungen
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=10),  # Besserer Sampler
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Pruning
        )

        study.optimize(objective, n_trials=30)  # 30 Trials für gute Balance

        # Beste Parameter loggen
        best_params = study.best_trial.params
        best_value = study.best_trial.value

        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_mse", best_value)
        mlflow.log_metric("n_trials", len(study.trials))
        mlflow.log_metric("n_completed_trials", len([t for t in study.trials if t.state.name == 'COMPLETE']))

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best MSE: {best_value:.6f}")
        print(f"Best parameters: {best_params}")

        # Save Optuna study
        study_path = PATHS.DATA_MODELS / "vecsets_regressor_optuna_study.joblib"
        joblib.dump(study, study_path)
        mlflow.log_artifact(str(study_path), "optuna_study")

        # Optuna Study Visualisierung (optional)
        try:
            import optuna.visualization as vis

            fig_importance = vis.plot_param_importances(study)
            fig_optimization = vis.plot_optimization_history(study)

            # Speichere Plots
            importance_path = PATHS.DATA_MODELS / "transformer_param_importance.html"
            history_path = PATHS.DATA_MODELS / "transformer_optimization_history.html"

            fig_importance.write_html(str(importance_path))
            fig_optimization.write_html(str(history_path))

            mlflow.log_artifact(str(importance_path), "plots")
            mlflow.log_artifact(str(history_path), "plots")

            print("Optuna visualizations saved and logged to MLflow!")
        except ImportError:
            print("Optuna visualization not available - install plotly for plots")

    print("\nTraining final model with best parameters...")

    
    with mlflow.start_run(run_name="final_best_transformer_model"):

        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "final_best_transformer_model")

        train_loader, val_loader, log_scaler = get_dataloaders_with_scaler(
            batch_size=best_params["batch_size"]
        )
        

        model = VecsetTransformerRegressor(
            input_dim=32,
            n_targets=4,
            embed_dim=best_params["embed_dim"],
            num_heads=best_params["num_heads"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            use_target_specific_heads=best_params["use_target_heads"],
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            max_epochs=300,  # Mehr Epochs für finales Training
            warmup_epochs=best_params["warmup_epochs"],
            log_scaler=log_scaler,
            target_names=['VOLUME', 'FACES', 'EDGES', 'VERTICES']
        )
        

        model_params = sum(p.numel() for p in model.parameters())
        mlflow.log_metric("model_parameters", model_params)
        

        mlf_logger = MLFlowLogger(
            experiment_name="vecset-regression",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id
        )
        
        # Callbacks 
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=40,  
            mode='min',
            verbose=False
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            dirpath=PATHS.DATA_MODELS.as_posix(),
            filename='vecset-regressor',
            save_weights_only=False,
            verbose=False
        )
        
        # final trainer
        trainer = Trainer(
            max_epochs=300,
            logger=mlf_logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            enable_checkpointing=True,
            enable_progress_bar=True,  
            gradient_clip_val=1.0,
            precision='16-mixed' if torch.cuda.is_available() else '32',
            accumulate_grad_batches=1,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            detect_anomaly=False  
        )
        
        
        trainer.fit(model, train_loader, val_loader)

        final_metrics = trainer.callback_metrics
        for key, value in final_metrics.items():
            if torch.is_tensor(value):
                mlflow.log_metric(f"final_{key}", value.item())
        

        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            mlflow.log_artifact(best_model_path, "model")
            print(f"Best model saved to: {best_model_path}")
        

        scaler_path = PATHS.DATA_MODELS / "transformer_log_scaler.joblib"
        best_params_path = PATHS.DATA_MODELS / "transformer_best_params.joblib"

        joblib.dump(log_scaler, scaler_path)
        joblib.dump(best_params, best_params_path)
        
        # Zu MLflow hinzufügen
        mlflow.log_artifact(str(scaler_path), "scaler")
        mlflow.log_artifact(str(best_params_path), "parameters")

        print(f"Scaler saved to: {scaler_path}")
        print(f"Best params saved to: {best_params_path}")
        print("Final training completed!")
        



def evaluate_transformer(model_path=None, scaler_path=None):
    """Separate Funktion für Transformer Evaluation auf Test-Set"""
    
    with mlflow.start_run(run_name="transformer_model_evaluation"):
        print("="*60)
        print("EVALUATING TRANSFORMER ON TEST SET")
        print("="*60)
        

        if scaler_path is None:
            scaler_path = PATHS.DATA_MODELS / "transformer_log_scaler.joblib"
        if model_path is None:
            model_files = list(PATHS.DATA_MODELS.glob("transformer-regressor.ckpt"))
            if not model_files:
                print("Error: No transformer model checkpoint found!")
                return
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("scaler_path", str(scaler_path))
        
        if not scaler_path.exists():
            print("Error: Scaler not found. Run training first!")
            return
        
        print(f"Loading model from: {model_path}")
        
        log_scaler = joblib.load(scaler_path)
        model = VecsetTransformerRegressor.load_from_checkpoint(
            model_path, 
            log_scaler=log_scaler,
            strict=False  
        )
        
        # Test Dataset
        test_dataset = FabwaveDataset(
            csv_file="/clear-shape/data/5_model_input/test.csv", 
            classification=False,
            regression=True,
            data_type="vecsets"
        )
        
        test_loader = DataLoader(
            LogTransformDataset(test_dataset, log_scaler, fit_scaler=False),
            batch_size=256, 
            shuffle=False,
            num_workers=4
        )
        

        mlf_logger = MLFlowLogger(
            experiment_name="vecsets-regression",
            tracking_uri=PATHS.MLFLOW_TRACKING_URI.as_posix(),
            run_id=mlflow.active_run().info.run_id
        )
        
        # Evaluation
        trainer = Trainer(
            logger=mlf_logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            precision='16-mixed' if torch.cuda.is_available() else '32'
        )
        
        results = trainer.test(model, test_loader)
        
        if results:
            for key, value in results[0].items():
                mlflow.log_metric(f"test_{key}", value)
        

        target_names = ['VOLUME', 'FACES', 'EDGES', 'VERTICES']
        
        
        return results[0] if results else None


def compare_models():
    """Vergleiche MLP vs Transformer Performance"""
    
    with mlflow.start_run(run_name="model_comparison"):

        mlp_study_path = PATHS.DATA_MODELS / "mlp_optuna_study.joblib"
        transformer_study_path = PATHS.DATA_MODELS / "transformer_optuna_study.joblib"
        
        results = {}
        
        if mlp_study_path.exists():
            mlp_study = joblib.load(mlp_study_path)
            results['MLP'] = {
                'best_mse': mlp_study.best_trial.value,
                'n_trials': len(mlp_study.trials),
                'best_params': mlp_study.best_trial.params
            }
        
        if transformer_study_path.exists():
            transformer_study = joblib.load(transformer_study_path)
            results['Transformer'] = {
                'best_mse': transformer_study.best_trial.value,
                'n_trials': len(transformer_study.trials),
                'best_params': transformer_study.best_trial.params
            }
        
        for model_name, data in results.items():
            mlflow.log_metric(f"{model_name.lower()}_best_mse", data['best_mse'])
            mlflow.log_metric(f"{model_name.lower()}_n_trials", data['n_trials'])
            

        
        if len(results) == 2:
            mse_diff = abs(results['MLP']['best_mse'] - results['Transformer']['best_mse'])
            better_model = 'MLP' if results['MLP']['best_mse'] < results['Transformer']['best_mse'] else 'Transformer'
            
            mlflow.log_metric("mse_difference", mse_diff)
            mlflow.log_param("better_model", better_model)
            

        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer Regression Training')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'compare'], 
                       default='train', help='Operation mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        main()
    elif args.mode == 'evaluate':
        evaluate_transformer()
    elif args.mode == 'compare':
        compare_models()
    else:
        # Für normale Ausführung ohne Argumente
        main()