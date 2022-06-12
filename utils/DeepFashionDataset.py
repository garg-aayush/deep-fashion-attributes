import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class DeepFashionDataset(Dataset):
    def __init__(self, df, image_dir='', transform=None):
        self.df = df
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = f"{self.image_dir}{self.df['filename'][index]}"
        #print(filename)
        image = cv2.imread(filename)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        label_neck = self.df['neck'][index]
        label_sleeve_length = self.df['sleeve_length'][index]
        label_pattern = self.df['pattern'][index]

        # image to float32 tensor
        image = image.type(torch.FloatTensor)
        # torch.tensor(image, dtype=torch.float32)

        # labels to int tensors
        label_neck = torch.tensor(label_neck, dtype=torch.long)
        label_sleeve_length = torch.tensor(label_sleeve_length, dtype=torch.long)
        label_pattern = torch.tensor(label_pattern, dtype=torch.long)

        return {
            'image': image,
            'neck': label_neck,
            'sleeve_length': label_sleeve_length,
            'pattern': label_pattern
        }
