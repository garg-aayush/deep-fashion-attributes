import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

from utils.DeepFashionDataset import DeepFashionDataset

IMG_DIR = '../data/classification-assignment/images/'
CSV_FILE = '../data/classification-assignment/attributes_processed.csv'

# load the csv file
df = pd.read_csv(CSV_FILE)

# total number of examples
nexamples = df.shape[0]
print(f'Total no of files: {nexamples}')
print(df.head())


image_size = 224

# basic transforms
transform_img = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), 
                ])

# set image folder path
image_dataset = DeepFashionDataset( df, IMG_DIR,transform_img)

# image_data_loader
image_data_loader = DataLoader(
  image_dataset, 
  batch_size=1,
  shuffle=False, 
  num_workers=4
)

# placeholders
psum    = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for idx, data in enumerate(image_data_loader):
    inputs = data['image']
    #print(inputs.shape)
    psum    += inputs.sum(axis        = [0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])


# pixel count
count = len(df) * image_size * image_size

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)

# output
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))

