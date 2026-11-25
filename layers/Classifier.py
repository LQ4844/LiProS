# Classifier provenance is listed under Acknowledgements in README.md.
import torch.nn as nn
import torch


class Classifier(nn.Module):

    def __init__(self, hidden_size, num_class):
        super().__init__()

        self.linear = nn.Linear(hidden_size, num_class, bias=True)
        self.reset_parameters()

    def forward(self, x):
        logits = self.linear(x)
        prediction = torch.argmax(logits, dim=1)

        return logits, prediction

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class DualClassifier(nn.Module):

    def __init__(self, hidden_size, num_class, mid_size=128, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_size, num_class),
        )
        self.reset_parameters()

    def forward(self, x):
        logits = self.mlp(x)
        prediction = torch.argmax(logits, dim=1)
        return logits, prediction

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
