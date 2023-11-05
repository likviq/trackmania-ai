import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

class ResNet18CustomHead(nn.Module):
    def __init__(self):
        super(ResNet18CustomHead, self).__init__()
        self.weights = ResNet18_Weights.DEFAULT
        resnet18 = models.resnet18(weights=self.weights)

        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, image):
        extracted_features = self.resnet18(image)

        predicted_input = self.head(extracted_features)

        return predicted_input