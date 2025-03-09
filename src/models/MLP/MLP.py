import torch.nn as nn

    

class MLP(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_dim:int=1024, num_layers:int=4):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  

        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers-2):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
            ))

        self.last_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )


    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.last_layer(x)
        return x