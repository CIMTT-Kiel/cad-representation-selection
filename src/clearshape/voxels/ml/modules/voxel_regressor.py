from typing import Optional, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from clearshape.models.cnn3d import VoxelRegressorCNN3D


class VoxelRegressor(pl.LightningModule):
    """PyTorch Lightning module for multi-target 3D CNN regression."""

    def __init__(self,
                 n_targets: int = 4,
                 dropout: float = 0.1,
                 use_target_specific_heads: bool = True,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-4,
                 log_scaler=None,
                 target_names: Optional[List[str]] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['log_scaler'])

        self.model = VoxelRegressorCNN3D(
            n_targets=n_targets,
            dropout=dropout,
            use_target_specific_heads=use_target_specific_heads,
        )

        self.log_scaler = log_scaler

        self.target_names = target_names or ['VOLUME', 'FACES', 'EDGES', 'VERTICES']

        for stage in ('train', 'val', 'test'):
            setattr(self, f'{stage}_mse', nn.ModuleDict({n: MeanSquaredError() for n in self.target_names}))
            setattr(self, f'{stage}_mae', nn.ModuleDict({n: MeanAbsoluteError() for n in self.target_names}))

    def forward(self, x):
        return self.model(x)

    def _compute_metrics(self, predictions, targets, stage='train'):
        metrics = {}
        for i, name in enumerate(self.target_names):
            mse = getattr(self, f'{stage}_mse')[name](predictions[:, i], targets[:, i])
            mae = getattr(self, f'{stage}_mae')[name](predictions[:, i], targets[:, i])
            metrics[f'{stage}_{name}_mse'] = mse
            metrics[f'{stage}_{name}_mae'] = mae
        metrics[f'{stage}_mse'] = F.mse_loss(predictions, targets)
        metrics[f'{stage}_mae'] = F.l1_loss(predictions, targets)
        return metrics

    def _compute_original_scale_metrics(self, predictions, targets, stage='val'):
        if self.log_scaler is None:
            return {}
        pred_np = predictions.detach().float().cpu().numpy()
        true_np = targets.detach().float().cpu().numpy()
        pred_orig = self.log_scaler.inverse_transform(pred_np)
        true_orig = self.log_scaler.inverse_transform(true_np)
        metrics = {}
        for i, name in enumerate(self.target_names):
            metrics[f'{stage}_{name}_mse_original'] = float(np.mean((pred_orig[:, i] - true_orig[:, i]) ** 2))
            metrics[f'{stage}_{name}_mae_original'] = float(np.mean(np.abs(pred_orig[:, i] - true_orig[:, i])))
        metrics[f'{stage}_mse_original'] = float(np.mean((pred_orig - true_orig) ** 2))
        metrics[f'{stage}_mae_original'] = float(np.mean(np.abs(pred_orig - true_orig)))
        return metrics

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        predictions = self(x)
        loss, individual_losses = self.model.compute_weighted_loss(predictions, y)

        # Variance calibration: penalize std over-amplification in log-space
        if predictions.size(0) > 1:
            var_ratio = predictions.std(dim=0) / (y.std(dim=0).detach() + 1e-6)
            var_loss = F.mse_loss(var_ratio, torch.ones_like(var_ratio))
            loss = loss + var_loss
            self.log('train_var_calibration', var_loss, on_step=False, on_epoch=True)

        metrics = self._compute_metrics(predictions, y, 'train')
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True)
        for i, name in enumerate(self.target_names):
            self.log(f'train_{name}_loss', individual_losses[i], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        predictions = self(x)
        loss, individual_losses = self.model.compute_weighted_loss(predictions, y)

        metrics = self._compute_metrics(predictions, y, 'val')
        original_metrics = self._compute_original_scale_metrics(predictions, y, 'val')

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True)
        for name, value in original_metrics.items():
            self.log(name, value, on_step=False, on_epoch=True,
                     prog_bar=('mse_original' in name and 'VOLUME' not in name))
        for i, name in enumerate(self.target_names):
            self.log(f'val_{name}_loss', individual_losses[i], on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        predictions = self(x)
        loss, _ = self.model.compute_weighted_loss(predictions, y)

        metrics = self._compute_metrics(predictions, y, 'test')
        original_metrics = self._compute_original_scale_metrics(predictions, y, 'test')

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
            pred_orig = self.log_scaler.inverse_transform(predictions_log.detach().cpu().numpy())
            result['predictions_original'] = torch.tensor(pred_orig)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"},
        }

    def _reset_metrics(self, stage):
        for metric_dict in [getattr(self, f'{stage}_mse'), getattr(self, f'{stage}_mae')]:
            for m in metric_dict.values():
                m.reset()

    def on_train_epoch_end(self):
        self._reset_metrics('train')

    def on_validation_epoch_end(self):
        self._reset_metrics('val')

    def on_test_epoch_end(self):
        self._reset_metrics('test')
