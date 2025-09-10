
# standard library imports
import logging

# third party imports
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)

class VecsetClassifier(nn.Module):
    def __init__(self, input_dim=32, d_model=1024, nhead=4, num_layers=4, num_classes=40,
                 dim_feedforward=512, dropout=0.1, fc_layers=None, use_pos_embedding=True):
        super().__init__()

        self.use_pos_embedding = use_pos_embedding

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model))
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Build classifier head dynamically
        layers = [nn.LayerNorm(d_model)]
        in_dim = d_model

        if fc_layers is not None:
            for hidden_dim in fc_layers:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_proj(x)
        if self.use_pos_embedding:
            x = x + self.pos_embedding
        encoded = self.encoder(x)
        cls_token = encoded[:, 0, :]
        out = self.classifier(cls_token)
        
        return out
    


class TransformerRegressor(nn.Module):
    """
    Transformer-based Multi-Target Regressor für log-transformierte Zielvariablen
    (VOLUME, FACES, EDGES, VERTICES)
    """
    
    def __init__(self, 
                 input_dim=16,  # Angepasst für Ihre Invarianten
                 n_targets=4,   # VOLUME, FACES, EDGES, VERTICES
                 embed_dim=256, # Kleiner für weniger Features
                 num_heads=8, 
                 num_layers=4, 
                 dropout=0.1,
                 use_target_specific_heads=True,
                 target_weights=None):
        super().__init__()
        
        self.n_targets = n_targets
        self.use_target_specific_heads = use_target_specific_heads
        
        # Target-spezifische Gewichte (basierend auf Log-Std)
        if target_weights is None:
            self.target_weights = torch.tensor([1/2.89, 1/1.06, 1/1.28, 1/1.28])
            self.target_weights = self.target_weights / self.target_weights.sum() * n_targets
        else:
            self.target_weights = torch.tensor(target_weights)
        
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 2,  # Standard: 2x embed_dim
            dropout=dropout, 
            batch_first=True,
            activation='gelu'  # GELU oft besser als ReLU für Transformer
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Feature Aggregation
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout für Regularisierung
        self.feature_dropout = nn.Dropout(dropout * 0.5)
        
        # Output Strategy
        if use_target_specific_heads:
            # Separate Köpfe für jedes Target
            self.target_heads = nn.ModuleList([
                self._create_target_head(embed_dim, f"target_{i}") 
                for i in range(n_targets)
            ])
        else:
            # Gemeinsamer Output Layer
            self.regressor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(embed_dim // 2, n_targets)
            )
        
        # Initialisierung
        self._init_weights()
    
    def _create_target_head(self, embed_dim, name):
        """Erstellt target-spezifischen Regression-Kopf"""
        return nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )
    
    def _init_weights(self):
        """Xavier/Kaiming Initialisierung"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        """
        Forward Pass
        Args:
            x: [B, input_dim] - Batch von Feature-Vektoren
        Returns:
            output: [B, n_targets] - Multi-Target Predictions
        """
        batch_size = x.size(0)
        
        # Input Projection
        x = self.input_projection(x)  # [B, input_dim, embed_dim]
        
        # Transformer Encoding
        x = self.encoder(x)  # [B, input_dim, embed_dim]
        
        # Global Pooling über Set-Dimension
        x = x.transpose(1, 2)  # [B, embed_dim, input_dim]
        x = self.pool(x).squeeze(-1)  # [B, embed_dim]
        
        # Feature Dropout
        x = self.feature_dropout(x)
        
        # Output Generation
        if self.use_target_specific_heads:
            outputs = []
            for head in self.target_heads:
                outputs.append(head(x))  # Jeder Kopf gibt [B, 1] aus
            output = torch.cat(outputs, dim=1)  # [B, n_targets]
        else:
            output = self.regressor(x)  # [B, n_targets]
        
        return output
    
    def compute_weighted_loss(self, predictions, targets, loss_fn=nn.MSELoss(reduction='none')):
        """Gewichteter Loss für Multi-Target Regression"""
        target_losses = loss_fn(predictions, targets)  # [B, n_targets]
        weights = self.target_weights.to(predictions.device)
        weighted_losses = target_losses * weights.unsqueeze(0)
        total_loss = weighted_losses.mean()
        individual_losses = target_losses.mean(dim=0)
        return total_loss, individual_losses
    
# Example usage
if __name__ == "__main__":
    model = VecsetClassifier()
    x = torch.randn(32, 1024, 32)  # Batch size of 32, sequence length of 1024, input dimension of 32
    output = model(x)
    print(output.shape)  # Should be [32, num_classes]