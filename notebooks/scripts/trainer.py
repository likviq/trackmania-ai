import sys
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CyclicLR
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

sys.path.append('data_preparation')
from data_preparation.dataset_prep_main import DatasetPreparation

sys.path.append('models')
from models.vgg16_with_custom_head import Vgg16CustomHead

class Trainer:
    def __init__(self, df_path):

        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor()])
        
        self.data_prep = DatasetPreparation(df_path=df_path, 
                                            transform=self.transform)
        
        self.train_loader, self.test_loader = self.data_prep.get_train_test_dataloader()

        self.model = Vgg16CustomHead()

        self.criterion = nn.MSELoss()

        self.num_epochs = 100

        self.lr = 0.0001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.wd = 0.01

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, num_epochs=None, lr=None, wd=None, freeze_epochs=10, lr_find=False, checkpoint_location=None):
        
        if num_epochs is None: num_epochs = self.num_epochs
        if lr is None: lr = self.lr
        if wd is None: wd = self.wd

        if checkpoint_location is not None:
            # TODO: implement uploading from checkpoints
            return
        
        train_loss = []

        for epoch in tqdm(range(num_epochs)):
            for batch in self.train_loader:
                image, speed, cur_gear, side_speed, input_value = batch
                image = image.to(torch.float32)
                input_value = input_value.to(torch.float32)
                image, speed, cur_gear, side_speed, input_value = image.to(self.device), speed.to(self.device), cur_gear.to(self.device), side_speed.to(self.device), input_value.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(image)
                # print((outputs, input_value))
                loss = self.criterion(outputs, input_value)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            train_loss.append(loss.item())