#third party imports
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from typing import Optional, List, Dict, Any

#custom imports
from clearshape.models.invariant_mlp import InvariantRegressorMLP

class InvariantRegressor(pl.LightningModule):
    """
    PyTorch Lightning Module für Multi-Target Regression
    """
    
    def __init__(self, 
                 in_dim: int,
                 n_targets: int = 4,
                 dropout: float = 0.15,
                 fc_layers: List[int] = [128, 256, 128, 64],
                 act_fn = nn.ReLU,
                 use_target_specific_heads: bool = True,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 log_scaler = None,
                 target_names: Optional[List[str]] = None):
        
        super().__init__()
        self.save_hyperparameters(ignore=['log_scaler'])
        
        # Model
        self.model = InvariantRegressorMLP(
            in_dim=in_dim,
            n_targets=n_targets,
            dropout=dropout,
            fc_layers=fc_layers,
            act_fn=act_fn,
            use_target_specific_heads=use_target_specific_heads
        )
        
        # Log Scaler für Evaluation auf ursprünglicher Skala
        self.log_scaler = log_scaler
        
        # Target Namen für Logging
        if target_names is None:
            self.target_names = ['VOLUME', 'FACES', 'EDGES', 'VERTICES']
        else:
            self.target_names = target_names
            
        # Metriken für jedes Target
        self.train_mse = nn.ModuleDict({
            name: MeanSquaredError() for name in self.target_names
        })
        self.val_mse = nn.ModuleDict({
            name: MeanSquaredError() for name in self.target_names
        })
        self.test_mse = nn.ModuleDict({
            name: MeanSquaredError() for name in self.target_names
        })
        
        self.train_mae = nn.ModuleDict({
            name: MeanAbsoluteError() for name in self.target_names
        })
        self.val_mae = nn.ModuleDict({
            name: MeanAbsoluteError() for name in self.target_names
        })
        self.test_mae = nn.ModuleDict({
            name: MeanAbsoluteError() for name in self.target_names
        })
        
        # Für Original-Scale Evaluation
        self.val_original_mse = {}
        self.test_original_mse = {}
        
    def forward(self, x):
        return self.model(x)
    
    def _compute_metrics(self, predictions, targets, stage='train'):
        """Berechnet Metriken für alle Targets"""
        metrics = {}
        
        # MSE und MAE pro Target (auf Log-Skala)
        for i, target_name in enumerate(self.target_names):
            pred_target = predictions[:, i]
            true_target = targets[:, i]
            
            mse_metric = getattr(self, f'{stage}_mse')[target_name]
            mae_metric = getattr(self, f'{stage}_mae')[target_name]
            
            mse_val = mse_metric(pred_target, true_target)
            mae_val = mae_metric(pred_target, true_target)
            
            metrics[f'{stage}_{target_name}_mse'] = mse_val
            metrics[f'{stage}_{target_name}_mae'] = mae_val
        
        # Overall MSE und MAE
        overall_mse = F.mse_loss(predictions, targets)
        overall_mae = F.l1_loss(predictions, targets)
        
        metrics[f'{stage}_mse'] = overall_mse
        metrics[f'{stage}_mae'] = overall_mae
        
        return metrics
    
    def _compute_original_scale_metrics(self, predictions, targets, stage='val'):
        """Berechnet Metriken auf ursprünglicher Skala"""
        if self.log_scaler is None:
            return {}
            
        # Zurück zur ursprünglichen Skala
        pred_np = predictions.detach().cpu().numpy()
        true_np = targets.detach().cpu().numpy()
        
        pred_original = self.log_scaler.inverse_transform(pred_np)
        true_original = self.log_scaler.inverse_transform(true_np)
        
        metrics = {}
        
        # MSE pro Target auf ursprünglicher Skala
        for i, target_name in enumerate(self.target_names):
            mse_original = np.mean((pred_original[:, i] - true_original[:, i]) ** 2)
            mae_original = np.mean(np.abs(pred_original[:, i] - true_original[:, i]))
            
            metrics[f'{stage}_{target_name}_mse_original'] = mse_original
            metrics[f'{stage}_{target_name}_mae_original'] = mae_original
        
        # Overall auf ursprünglicher Skala
        overall_mse_original = np.mean((pred_original - true_original) ** 2)
        overall_mae_original = np.mean(np.abs(pred_original - true_original))
        
        metrics[f'{stage}_mse_original'] = overall_mse_original
        metrics[f'{stage}_mae_original'] = overall_mae_original
        
        return metrics
    
    def training_step(self, batch, batch_idx):
        x, y = batch[:2]  # Falls batch mehr als x, y enthält
        
        # Forward pass
        predictions = self(x)
        
        # Gewichteter Loss
        loss, individual_losses = self.model.compute_weighted_loss(predictions, y)
        
        # Metriken berechnen
        metrics = self._compute_metrics(predictions, y, 'train')
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True)
        
        # Individual losses pro Target
        for i, target_name in enumerate(self.target_names):
            self.log(f'train_{target_name}_loss', individual_losses[i], 
                    on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        
        predictions = self(x)
        loss, individual_losses = self.model.compute_weighted_loss(predictions, y)
        
        # Metriken auf Log-Skala
        metrics = self._compute_metrics(predictions, y, 'val')
        
        # Metriken auf ursprünglicher Skala
        original_metrics = self._compute_original_scale_metrics(predictions, y, 'val')
        
        # Logging
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # Log-Skala Metriken
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True)
        
        # Original-Skala Metriken (wichtiger für Interpretation!)
        for name, value in original_metrics.items():
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=('mse_original' in name))
        
        # Individual losses
        for i, target_name in enumerate(self.target_names):
            self.log(f'val_{target_name}_loss', individual_losses[i], 
                    on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        
        predictions = self(x)
        loss, individual_losses = self.model.compute_weighted_loss(predictions, y)
        
        # Metriken
        metrics = self._compute_metrics(predictions, y, 'test')
        original_metrics = self._compute_original_scale_metrics(predictions, y, 'test')
        
        # Logging
        self.log('test_loss', loss, prog_bar=True)
        
        for name, value in metrics.items():
            self.log(name, value)
            
        for name, value in original_metrics.items():
            self.log(name, value, prog_bar=('mse_original' in name))
        
        return loss
    
    def predict_step(self, batch, batch_idx=None):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        predictions_log = self(x)
        
        # Rücktransformation zur ursprünglichen Skala
        if self.log_scaler is not None:
            predictions_original = self.log_scaler.inverse_transform(
                predictions_log.detach().cpu().numpy()
            )
            return {
                'predictions_log': predictions_log,
                'predictions_original': torch.tensor(predictions_original)
            }
        else:
            return {'predictions_log': predictions_log}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=1000,  # Passen Sie das an Ihre Epoch-Anzahl an
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }
    
    def on_train_epoch_end(self):
        """Reset metrics at end of epoch"""
        for metric_dict in [self.train_mse, self.train_mae]:
            for metric in metric_dict.values():
                metric.reset()
    
    def on_validation_epoch_end(self):
        """Reset metrics at end of epoch"""
        for metric_dict in [self.val_mse, self.val_mae]:
            for metric in metric_dict.values():
                metric.reset()
                
    def on_test_epoch_end(self):
        """Reset metrics at end of epoch"""
        for metric_dict in [self.test_mse, self.test_mae]:
            for metric in metric_dict.values():
                metric.reset()


# Factory function für einfache Erstellung
def create_invariant_regressor(in_dim: int, 
                              log_scaler=None,
                              **kwargs) -> InvariantRegressor:
    """
    Factory function für InvariantRegressor mit optimalen Defaults
    """
    defaults = {
        'n_targets': 4,
        'dropout': 0.15,
        'fc_layers': [128, 256, 128, 64],
        'act_fn': nn.ReLU,
        'use_target_specific_heads': True,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'target_names': ['VOLUME', 'FACES', 'EDGES', 'VERTICES']
    }
    
    # Update defaults with provided kwargs
    defaults.update(kwargs)
    
    return InvariantRegressor(
        in_dim=in_dim,
        log_scaler=log_scaler,
        **defaults
    )