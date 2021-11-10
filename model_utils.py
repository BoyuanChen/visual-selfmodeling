
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)

class StateConditionMLPQueryModel(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=1, hidden_features=256):
        super(StateConditionMLPQueryModel, self).__init__()

        half_hidden_features = int(hidden_features / 2)
        self.layerq1 = SirenLayer(3, half_hidden_features, is_first=True)
        self.layers1 = SirenLayer(in_channels-3, half_hidden_features, is_first=True)
        self.layers2 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers3 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers4 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layer2 = SirenLayer(hidden_features, hidden_features)
        self.layer3 = SirenLayer(hidden_features, hidden_features)
        self.layer4 = SirenLayer(hidden_features, hidden_features)
        self.layer5 = SirenLayer(hidden_features, out_channels, is_last=True)
    
    def query_encoder(self, x):
        x = self.layerq1(x)
        return x

    def state_encoder(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        return x

    def forward(self, x):
        query_feat = self.query_encoder(x[:, :3])
        state_feat = self.state_encoder(x[:, 3:])
        x = torch.cat((query_feat, state_feat), dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class KinematicFeatToLinkModel(torch.nn.Module):
    def __init__(self, in_channels=128, out_channels=3, hidden_features=64):
        super(KinematicFeatToLinkModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, hidden_features)
        self.layer2 = SirenLayer(hidden_features, out_channels, is_last=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class KinematicScratchModel(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=3, hidden_features=128, hidden_hidden_features=64):
        super(KinematicScratchModel, self).__init__()

        # original self-model's kinematic branch
        self.layer1 = SirenLayer(in_channels, hidden_features, is_first=True)
        self.layer2 = SirenLayer(hidden_features, hidden_features)
        self.layer3 = SirenLayer(hidden_features, hidden_features)
        self.layer4 = SirenLayer(hidden_features, hidden_features)
        # newly added branches for X_link tasks
        self.layer5 = SirenLayer(hidden_features, hidden_hidden_features)
        self.layer6 = SirenLayer(hidden_hidden_features, out_channels, is_last=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x