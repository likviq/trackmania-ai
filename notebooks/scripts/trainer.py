import sys
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CyclicLR
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import v2

sys.path.append('data_preparation')
from data_preparation.dataset_prep_main import DatasetPreparation

sys.path.append('models')
from models.vgg16_with_custom_head import Vgg16CustomHead
from models.resnet18_with_custom_head import ResNet18CustomHead

import wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="trackmania-ai",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "Vgg16",
    "image_type": "RGB normalized",
    "dataset": "3-min-trackmania-map",
    "epochs": 100,
    }
)

class Trainer:
    def __init__(self, df_path):

        self.transform = v2.Compose([v2.Resize((224, 224)),
                                     v2.ToTensor(),
                                     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.data_prep = DatasetPreparation(df_path=df_path, 
                                            transform=self.transform)
        
        self.train_loader, self.test_loader = self.data_prep.get_train_test_dataloader()

        self.model = Vgg16CustomHead()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        self.num_epochs = 100

        self.wd = 0.01
        self.lr = 0.0001
        self.max_lr = 0.01
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr=self.lr, max_lr=self.max_lr, 
                                                     step_size_up=2000, step_size_down=None, 
                                                     mode='triangular', gamma=1.0,cycle_momentum=False)


    def train(self, num_epochs=None, lr=None, wd=None, freeze_epochs=10, lr_find=False, checkpoint_location=None):
        
        if num_epochs is None: num_epochs = self.num_epochs
        if lr is None: lr = self.lr
        if wd is None: wd = self.wd

        if checkpoint_location is not None:
            # TODO: implement uploading from checkpoints
            return
        
        train_loss = []

        for epoch in tqdm(range(num_epochs)):
            running_mse_loss, running_mae_loss, running_loss = 0, 0, 0
            batch_count = 0
            total_samples = 0
            for batch in self.train_loader:
                image, speed, cur_gear, side_speed, input_value = batch
                image = image.to(torch.float32)
                input_value = input_value.to(torch.float32)
                image, speed, cur_gear, side_speed, input_value = image.to(self.device), speed.to(self.device), cur_gear.to(self.device), side_speed.to(self.device), input_value.to(self.device)

                outputs = self.model(image)
                mse_loss = self.mse_criterion(outputs, input_value.view(len(input_value), 1))
                mae_loss = self.mae_criterion(outputs, input_value.view(len(input_value), 1))
                loss = (mse_loss + mae_loss)
                
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                running_mse_loss += mse_loss.item()
                running_mae_loss += mae_loss.item()
                batch_count += 1

                total_samples += input_value.size(0)

            if epoch % 10 == 0:
                print(outputs)
                print()
                print(input_value)

            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            avegare_loss = running_loss / batch_count
            avegare_mse_loss = running_mse_loss / batch_count
            avegare_mae_loss = running_mae_loss / batch_count

            wandb.log({"training_loss": running_loss, 
                       "learning_rate": current_lr, 
                       "Average_loss": avegare_loss,
                       "Average_MSE_loss": avegare_mse_loss,
                       "Average_MAE_loss": avegare_mae_loss})
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avegare_loss:.4f}')
            train_loss.append(loss.item())
        
        wandb.watch(self.model)