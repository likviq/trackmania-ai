import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights

class Vgg16CustomHead(nn.Module):
    def __init__(self):
        super(Vgg16CustomHead, self).__init__()
        self.weights = VGG16_Weights.DEFAULT
        vgg16 = models.vgg16(weights=self.weights)

        self.vgg16 = nn.Sequential(*list(vgg16.children())[:-1])
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, image):
        extracted_features = self.vgg16(image)

        predicted_input = self.head(extracted_features)

        return predicted_input