# standard library imports
import logging
from typing import Optional, List

# third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# custom imports
from clearshape.models.trnsfm_encoder import TransformerRegressor

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


class VecsetTransformerRegressor(pl.LightningModule):
    """
    PyTorch Lightning Module für Multi-Target Transformer Regression
    """
    
    def __init__(self,
                 input_dim: int = 16,
                 n_targets: int = 4,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 use_target_specific_heads: bool = True,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 max_epochs: int = 200,
                 warmup_epochs: int = 10,
                 log_scaler=None,
                 target_names: Optional[List[str]] = None):
        
        super().__init__()
        self.save_hyperparameters(ignore=['log_scaler'])
        
        # Model
        self.model = TransformerRegressor(
            input_dim=input_dim,
            n_targets=n_targets,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_target_specific_heads=use_target_specific_heads
        )
        
        # Log Scaler für Original-Scale Evaluation
        self.log_scaler = log_scaler
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        
        # Target Namen
        if target_names is None:
            self.target_names = ['VOLUME', 'FACES', 'EDGES', 'VERTICES']
        else:
            self.target_names = target_names
        
        # Metriken
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
    
    def forward(self, x):
        return self.model(x)
    
    def _compute_metrics(self, predictions, targets, stage='train'):
        """Berechnet Metriken für alle Targets"""
        metrics = {}
        
        for i, target_name in enumerate(self.target_names):
            pred_target = predictions[:, i]
            true_target = targets[:, i]
            
            mse_metric = getattr(self, f'{stage}_mse')[target_name]
            mae_metric = getattr(self, f'{stage}_mae')[target_name]
            
            mse_val = mse_metric(pred_target, true_target)
            mae_val = mae_metric(pred_target, true_target)
            
            metrics[f'{stage}_{target_name}_mse'] = mse_val
            metrics[f'{stage}_{target_name}_mae'] = mae_val
        
        # Overall Metriken
        overall_mse = F.mse_loss(predictions, targets)
        overall_mae = F.l1_loss(predictions, targets)
        
        metrics[f'{stage}_mse'] = overall_mse
        metrics[f'{stage}_mae'] = overall_mae
        
        return metrics
    
    def _compute_original_scale_metrics(self, predictions, targets, stage='val'):
        """Metriken auf ursprünglicher Skala"""
        if self.log_scaler is None:
            return {}
        
        pred_np = predictions.detach().cpu().numpy()
        true_np = targets.detach().cpu().numpy()
        
        pred_original = self.log_scaler.inverse_transform(pred_np)
        true_original = self.log_scaler.inverse_transform(true_np)
        
        metrics = {}
        
        for i, target_name in enumerate(self.target_names):
            mse_original = np.mean((pred_original[:, i] - true_original[:, i]) ** 2)
            mae_original = np.mean(np.abs(pred_original[:, i] - true_original[:, i]))
            
            metrics[f'{stage}_{target_name}_mse_original'] = mse_original
            metrics[f'{stage}_{target_name}_mae_original'] = mae_original
        
        overall_mse_original = np.mean((pred_original - true_original) ** 2)
        overall_mae_original = np.mean(np.abs(pred_original - true_original))
        
        metrics[f'{stage}_mse_original'] = overall_mse_original
        metrics[f'{stage}_mae_original'] = overall_mae_original
        
        return metrics
    
    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        
        predictions = self(x)
        loss, individual_losses = self.model.compute_weighted_loss(predictions, y)
        
        # Metriken
        metrics = self._compute_metrics(predictions, y, 'train')
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True)
        
        # Individual Target Losses
        for i, target_name in enumerate(self.target_names):
            self.log(f'train_{target_name}_loss', individual_losses[i], 
                    on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]
        
        predictions = self(x)
        loss, individual_losses = self.model.compute_weighted_loss(predictions, y)
        
        # Log-Scale Metriken
        metrics = self._compute_metrics(predictions, y, 'val')
        
        # Original-Scale Metriken
        original_metrics = self._compute_original_scale_metrics(predictions, y, 'val')
        
        # Logging
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True)
        
        for name, value in original_metrics.items():
            self.log(name, value, on_step=False, on_epoch=True, 
                    prog_bar=('mse_original' in name and 'VOLUME' not in name))
        
        # Individual Losses
        for i, target_name in enumerate(self.target_names):
            self.log(f'val_{target_name}_loss', individual_losses[i], 
                    on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch[:2]
        
        predictions = self(x)
        loss, individual_losses = self.model.compute_weighted_loss(predictions, y)
        
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
        
        result = {'predictions_log': predictions_log}
        
        if self.log_scaler is not None:
            predictions_original = self.log_scaler.inverse_transform(
                predictions_log.detach().cpu().numpy()
            )
            result['predictions_original'] = torch.tensor(predictions_original)
        
        return result
    
    def configure_optimizers(self):
        """Optimizers mit Warmup und Cosine Annealing"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95)  # Bessere Betas für Transformer
        )
        
        # Learning Rate Schedule mit Warmup
        def lr_lambda(current_step):
            if current_step < self.warmup_epochs:
                return float(current_step) / float(max(1, self.warmup_epochs))
            else:
                progress = float(current_step - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }
    
    def on_train_epoch_end(self):
        """Reset metrics"""
        for metric_dict in [self.train_mse, self.train_mae]:
            for metric in metric_dict.values():
                metric.reset()
    
    def on_validation_epoch_end(self):
        """Reset metrics"""
        for metric_dict in [self.val_mse, self.val_mae]:
            for metric in metric_dict.values():
                metric.reset()
    
    def on_test_epoch_end(self):
        """Reset metrics"""
        for metric_dict in [self.test_mse, self.test_mae]:
            for metric in metric_dict.values():
                metric.reset()


# Factory Function
def create_transformer_regressor(input_dim: int = 16,
                                log_scaler=None,
                                **kwargs) -> TransformerRegressor:
    """
    Factory für optimales Transformer-Regressionsmodell
    """
    defaults = {
        'n_targets': 4,
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'use_target_specific_heads': True,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'max_epochs': 200,
        'warmup_epochs': 10,
        'target_names': ['VOLUME', 'FACES', 'EDGES', 'VERTICES']
    }
    
    defaults.update(kwargs)
    
    return TransformerRegressor(
        input_dim=input_dim,
        log_scaler=log_scaler,
        **defaults
    )