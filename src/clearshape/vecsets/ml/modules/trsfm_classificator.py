import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.classification import Accuracy
#custom imports
from clearshape.models.vecset_trnsf_encoder import VecsetClassifier


class VecsetClassifierModule(pl.LightningModule):
    def __init__(self, 
                 lr = 1e-3,
                 input_dim=32, 
                 d_model=1024, 
                 nhead=8, 
                 num_layers=4, 
                 num_classes=40, 
                 dim_feedforward=2048, 
                 fc_layers=None,
                 dropout=0.3, 
                 use_pos_embedding=True, 
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.use_pos_embbeding = use_pos_embedding

        self.model = VecsetClassifier(
                                    input_dim=input_dim, 
                                    d_model=d_model, 
                                    nhead=nhead,
                                    num_layers=num_layers, 
                                    fc_layers=fc_layers,
                                    num_classes=num_classes,
                                    dim_feedforward=dim_feedforward, 
                                    dropout=dropout, 
                                    use_pos_embedding=use_pos_embedding
                                    )
        
        #metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # criterion
        self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, vector_set):
        return self.model(vector_set)

    def training_step(self, batch, batch_idx):
        vector_set, target_cls = batch

        # Convert one-hot to label indices
        if target_cls.ndim > 1 and target_cls.shape[-1] > 1:
            target_cls = torch.argmax(target_cls, dim=-1)

        logits = self(vector_set)
        loss = self.criterion(logits, target_cls)
        preds = torch.argmax(logits, dim=-1)

        self.train_acc(preds, target_cls)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vector_set, target_cls = batch

        if target_cls.ndim > 1 and target_cls.shape[-1] > 1:
            target_cls = torch.argmax(target_cls, dim=-1)

        logits = self(vector_set)
        val_loss = self.criterion(logits, target_cls)
        preds = torch.argmax(logits, dim=-1)

        self.val_acc(preds, target_cls)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])

    # Test Step anpassen
    def test_step(self, batch, batch_idx):
        vector_set, target_cls = batch

        # Wenn One-Hot-Labels verwendet werden, in Integer-Labels umwandeln
        if target_cls.ndim > 1 and target_cls.shape[-1] > 1:
            target_cls = torch.argmax(target_cls, dim=-1)

        logits = self(vector_set)
        test_loss = self.criterion(logits, target_cls)
        preds = torch.argmax(logits, dim=-1)

        # Metriken berechnen
        self.log("test_loss", test_loss)
        self.log("test_acc", self.val_acc(preds, target_cls))
        return test_loss

    # Predict Step anpassen
    def predict_step(self, batch):
        vector_set, _ = batch  # Kein Label nötig
        logits = self(vector_set)
        return torch.argmax(logits, dim=-1)


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
    
#test
if __name__ == "__main__":
    # Example usage
    model = VecsetClassifierModule()
    x = torch.randn(32, 1024, 32)  # Batch size of 32, sequence length of 1024, input dimension of 32
    output = model(x)
    print(output.shape)  # Should be [32, num_classes]