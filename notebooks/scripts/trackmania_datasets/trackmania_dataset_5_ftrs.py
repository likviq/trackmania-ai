from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class TrackmaniaAIDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        img_path = self.dataframe.iloc[idx]['image_path']
        speed = self.dataframe.iloc[idx]['speed:']
        cur_gear = self.dataframe.iloc[idx]['curGear:']
        side_speed = self.dataframe.iloc[idx]['sideSpeed:']
        input_value = self.dataframe.iloc[idx]['inputSteer:']
        
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)

        # input_value = round(input_value, 1)
        
        return image, speed, cur_gear, side_speed, input_value