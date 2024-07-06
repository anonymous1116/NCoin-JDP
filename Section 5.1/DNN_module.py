import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, D_in, D_out,H = 128, H2 = 128, H3 = 128, p1=0.2, p2=0.2, p3=0.2):
        super(Net, self).__init__()
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
