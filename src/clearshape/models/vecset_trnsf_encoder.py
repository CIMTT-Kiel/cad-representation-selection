import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
                layers.append(nn.ReLU())
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
    
# Example usage
if __name__ == "__main__":
    model = VecsetClassifier()
    x = torch.randn(32, 1024, 32)  # Batch size of 32, sequence length of 1024, input dimension of 32
    output = model(x)
    print(output.shape)  # Should be [32, num_classes]