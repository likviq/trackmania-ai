import torch.nn as nn
import torchvision.models as models

class Vgg16CustomHead(nn.Module):
    def __init__(self):
        super(Vgg16CustomHead, self).__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.vgg16 = nn.Sequential(*list(vgg16.children())[:-1])
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1)
        )

    def forward(self, image):
        extracted_features = self.vgg16(image)

        predicted_input = self.head(extracted_features)

        return predicted_input