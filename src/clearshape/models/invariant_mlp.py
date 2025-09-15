import torch
from torch import nn

class InvariantMLP(nn.Module):
    def __init__(self,
                in_dim=16,
                num_classes=38,
                dropout=0.0,
                fc_layers=[64,128,64],
                act_fn=nn.LeakyReLU,
                ):
        
        super().__init__()

        layers = []

        if fc_layers is not None:
            for hidden_dim in fc_layers:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim)),
                layers.append(act_fn())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out
    



class InvariantRegressorMLP(nn.Module):
    """
    Optimiertes MLP für Multi-Target Regression
    """
    
    def __init__(self,
                 in_dim=16,
                 n_targets=4,
                 dropout=0.1,
                 fc_layers=[128, 256, 128, 64],
                 act_fn=nn.ReLU,
                 use_residual=True,
                 use_target_specific_heads=True,
                 target_weights=None):
        super().__init__()
        
        self.n_targets = n_targets
        self.use_residual = use_residual
        self.use_target_specific_heads = use_target_specific_heads
        
        # Target-spezifische Gewichte
        if target_weights is None:
            # Gewichte basierend auf Log-Std: VOLUME=2.89, FACES=1.06, EDGES=1.28, VERTICES=1.28
            self.target_weights = torch.tensor([1/2.89, 1/1.06, 1/1.28, 1/1.28])
            self.target_weights = self.target_weights / self.target_weights.sum() * n_targets
        else:
            self.target_weights = torch.tensor(target_weights)
        
        # Shared Feature Extraction
        self.shared_layers = nn.ModuleList()
        current_dim = in_dim
        
        for i, hidden_dim in enumerate(fc_layers):
            self.shared_layers.append(nn.Linear(current_dim, hidden_dim))
            self.shared_layers.append(nn.BatchNorm1d(hidden_dim))
            self.shared_layers.append(act_fn())
            
            if i < len(fc_layers) - 1:
                self.shared_layers.append(nn.Dropout(dropout))
            
            current_dim = hidden_dim
        
        # Output Strategy
        if use_target_specific_heads:
            self.target_heads = nn.ModuleList([
                self._create_target_head(current_dim) 
                for _ in range(n_targets)
            ])
        else:
            self.output_layer = nn.Linear(current_dim, n_targets)
            
        self._init_weights()
        
    def _create_target_head(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(in_dim // 2, 1)
        )
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = x
        for layer in self.shared_layers:
            features = layer(features)
        
        if self.use_target_specific_heads:
            outputs = []
            for head in self.target_heads:
                outputs.append(head(features))
            output = torch.cat(outputs, dim=1)
        else:
            output = self.output_layer(features)
        
        return output
    
    def compute_weighted_loss(self, predictions, targets, loss_fn=nn.MSELoss(reduction='none')):
        target_losses = loss_fn(predictions, targets)
        weights = self.target_weights.to(predictions.device)
        weighted_losses = target_losses * weights.unsqueeze(0)
        total_loss = weighted_losses.mean()
        individual_losses = target_losses.mean(dim=0)
        return total_loss, individual_losses




# test

if __name__ == "__main__":
    model = InvariantMLP(in_dim=16, num_classes=38, dropout=0.0, fc_layers=[64, 128, 64])
    x = torch.randn(32, 16)  # batch size of 32 and input dimension of 16
    output = model(x)
    print(output.shape)  # should be (32, 38)