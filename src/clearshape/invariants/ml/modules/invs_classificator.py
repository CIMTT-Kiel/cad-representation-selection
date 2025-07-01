import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import F1Score

from clearshape.models.invariant_mlp import InvariantMLP

class InvariantClassifier(pl.LightningModule):
    def __init__(self, in_dim, num_classes, dropout=0.2, fc_layers=[64,128,64], act_fn=nn.LeakyReLU, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = InvariantMLP(in_dim=in_dim, num_classes=num_classes, dropout=dropout, fc_layers=fc_layers, act_fn=act_fn)

        #metrics
        self.f1_score = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        f1 = self.f1_score(logits.argmax(dim=1), y)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', (logits.argmax(dim=1) == y).float().mean(), prog_bar=True)
        self.log('train_f1_score', f1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)

        acc = (logits.argmax(dim=1) == y).float().mean()
        loss = F.cross_entropy(logits, y)

        f1_score = self.f1_score(logits.argmax(dim=1), y)

        self.log('val_acc', acc, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1_score', f1_score, prog_bar=True)

        return acc
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_acc', acc, prog_bar=True)
        return acc
    
    def predict_step(self, batch):
        x, y, _ = batch
        logits = self(x)
        return logits.argmax(dim=1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,          # Erste Periode (z. B. 10 Epochen)
            T_mult=2,        # Jede Periode wird doppelt so lang
            eta_min=1e-6     # Minimale Lernrate am Ende eines Zyklus
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",     # oder "step", je nachdem wie du willst
                "monitor": "val_loss",   # optional, z. B. bei ReduceLROnPlateau
            }
        }