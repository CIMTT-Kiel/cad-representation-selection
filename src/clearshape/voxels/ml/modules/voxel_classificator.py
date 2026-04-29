import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import F1Score

from clearshape.models.cnn3d import VoxelClassifierCNN3D


class VoxelClassifier(pl.LightningModule):
    def __init__(self, num_classes: int, dropout: float = 0.1, lr: float = 1e-4, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = VoxelClassifierCNN3D(num_classes=num_classes, dropout=dropout)
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        f1 = self.f1_score(logits.argmax(dim=1), y)

        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_f1_score', f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        f1 = self.f1_score(logits.argmax(dim=1), y)

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_f1_score', f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        f1 = self.f1_score(logits.argmax(dim=1), y)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1_score', f1, prog_bar=True)
        return acc

    def predict_step(self, batch, batch_idx=None):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x).argmax(dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"},
        }
