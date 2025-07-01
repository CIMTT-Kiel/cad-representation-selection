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
    


# test

if __name__ == "__main__":
    model = InvariantMLP(in_dim=16, num_classes=38, dropout=0.0, fc_layers=[64, 128, 64])
    x = torch.randn(32, 16)  # batch size of 32 and input dimension of 16
    output = model(x)
    print(output.shape)  # should be (32, 38)