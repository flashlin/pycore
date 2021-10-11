import torch
from sklearn.metrics import accuracy_score
from torch import nn

from ptorch.model_utils import Flatten


class CnnModel(nn.Module):

    def __init__(self):
        super(CnnModel).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten()
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.cnn_layers(x)
        # x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        return x

    def compute_loss(self, logits, y):
        return self.loss_func(logits, y)

    def accuracy_score(self, logits, y):
        _, y_hat = torch.max(logits, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), y.cpu())
        val_acc = torch.tensor(val_acc)
        return val_acc
