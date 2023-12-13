import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights


class YodaModel(nn.Module):
    def __init__(self, num_classes, weights=ResNet18_Weights.DEFAULT):
        super(YodaModel, self).__init__()

        self.num_classes = num_classes
        self.weights = weights

        # Initialize the resnet model with the default pretrained weights
        model = resnet18(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(model.fc.in_features, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.model(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return x
