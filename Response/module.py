import torch
import torch.nn as nn
import torch.nn.functional as F

class FL_Net(nn.Module):
    def __init__(self, D_in, D_out,H = 128, H2 = 128, H3 = 128, p1=0.1, p2=0.1, p3=0.1, device="cuda"):
        super().__init__()
        self.device = device
        
        self.fc1 = nn.Linear(D_in, H)
        self.bn1 = nn.BatchNorm1d(num_features=H)
        self.dn1 = nn.Dropout(p1)

        self.fc2 = nn.Linear(H, H2)
        self.bn2 = nn.BatchNorm1d(num_features=H2)
        self.dn2 = nn.Dropout(p2)

        self.fc3 = nn.Linear(H2, H3)
        self.bn3 = nn.BatchNorm1d(num_features=H3)
        self.dn3 = nn.Dropout(p3)

        self.fc4 = nn.Linear(H3, D_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
    

class GRU_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        GRU-based model for reducing a sequence to a fixed-size continuous output.
        Args:
            input_dim: Dimension of the input features at each time step.
            hidden_dim: Dimension of the GRU hidden state.
            output_dim: Dimension of the output (e.g., 3 for continuous variables).
        """
        super(GRU_net, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Map hidden state to final output78

    def forward(self, x):
        """
        Forward pass for the GRU model.
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
        Returns:
            Tensor of shape [batch_size, output_dim].
        """
        # Pass the sequence through the GRU
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # Now: (batch_size, 250, 1)
        
        _, hidden_state = self.gru(x)  # hidden_state: [1, batch_size, hidden_dim]
        
        # Use the last hidden state
        hidden_state = hidden_state.squeeze(0)  # [batch_size, hidden_dim]
        
        # Map to final output
        output = self.fc(hidden_state)  # [batch_size, output_dim]
        return output
