from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from PIL import Image
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from trackmania_datasets.trackmania_dataset_5_ftrs import TrackmaniaAIDataset

class DatasetPreparation():
    def __init__(self, df_path, transform):
        df = pd.read_csv(df_path)

        self.transform = transform
        self.df = self.remove_rows_without_images(df)


    def remove_rows_without_images(self, _df: pd.DataFrame):
        for index, row in _df.iterrows():
            if not os.path.exists(row['image_path']):
                _df.drop(index, inplace=True)

        return _df
    
    
    def get_train_test_dataloader(self):
        train_df, test_df = train_test_split(self.df, test_size=0.2, shuffle=False)

        train_dataset = TrackmaniaAIDataset(dataframe=train_df, transform=self.transform)
        test_dataset = TrackmaniaAIDataset(dataframe=test_df, transform=self.transform)

        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader