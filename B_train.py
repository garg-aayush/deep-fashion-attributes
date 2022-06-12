import pandas as pd
import torch
import torch.optim as optim
from utils.helper import train_val_split, save_model, save_loss_plot, loss_fn, set_random_seed, file_path, dir_path
from torch.utils.data import DataLoader
from utils.models import MultiHeadResNet18
from tqdm import tqdm
from torchvision import transforms
from utils.DeepFashionDataset import DeepFashionDataset
import torch.nn as nn
import argparse

# IMAGENET mean, std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Image size
IMAGE_SIZE = 224

# training function
def train(model, dataloader, optimizer, loss_fn, dataset, device):
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        #print(data.keys())
        # extract the features and labels
        image = data['image'].to(device)
        print
        neck = data['neck'].to(device)
        sleeve_length = data['sleeve_length'].to(device)
        pattern = data['pattern'].to(device)
        
        # zero-out the optimizer gradients
        optimizer.zero_grad()
        
        o1, o2, o3 = model(image)

        # print(o1.shape, neck.shape)
        # print(o2.shape, sleeve_length.shape)
        # print(o3.shape, pattern.shape)
        outputs = (o1, o2, o3)
        targets = (neck, sleeve_length, pattern)
        #print(targets)
        loss = loss_fn(outputs, targets)
        train_running_loss += loss.item()
        
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    return train_loss

# validation function
def validate(model, dataloader, loss_fn, dataset, device):
    model.eval()
    counter = 0
    val_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        # extract the features and labels
        image = data['image'].to(device)
        neck = data['neck'].to(device)
        sleeve_length = data['sleeve_length'].to(device)
        pattern = data['pattern'].to(device)
        
        outputs = model(image)
        targets = (neck, sleeve_length, pattern)
        loss = loss_fn(outputs, targets)
        val_running_loss += loss.item()
        
    val_loss = val_running_loss / counter
    return val_loss


def main():
    ######################################################################
    # input cmdline arguments
    ######################################################################
    parser=argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--run_name',default='test',required=True,type=str)
    parser.add_argument('--random_seed',default=42,type=int)             
    parser.add_argument('-ep','--epochs',default=20,type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('-bs','--batch_size',default=24,type=int)
    parser.add_argument('-nw','--num_workers',default=4,type=int)
    parser.add_argument('-lr', "--learning_rate", default=0.001, type=float,
                        help="Learning rate for Adam.")
    # paths
    parser.add_argument("--img_dir", default='../data/classification-assignment/images/', type=dir_path,
                        help="Path to input images folder")
    parser.add_argument("--csv_path", default='../data/classification-assignment/attributes_processed.csv', type=file_path,
                        help="Path to input images attributes file")
    parser.add_argument("--checkpoint_path", default='./outputs/model.pth', help="Path to save model")
    parser.add_argument("--lossgraph_path", default='./outputs/loss.jpg', help="Path to save loss graph")
        
    args=parser.parse_args()

    # print input arguments
    print(f'Run name : {args.run_name}')
    print(f'Random seed: {args.random_seed}')
    
    print(f'Num of training epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Num workers: {args.num_workers}')
    
    print(f'Learning rate: {args.learning_rate}')
    
    print(f'Image dir: {args.img_dir}')
    print(f'Csv path: {args.csv_path}')
    print(f'Checkpoint path: {args.checkpoint_path}')
    print(f'Lossgraph_path: {args.lossgraph_path}')

    
    ######################################################################
    # Set the random seed
    ######################################################################
    print(f'Set the random seed: {args.random_seed}')
    set_random_seed(args.random_seed)
    
    
    ######################################################################
    # Input train/val data
    ######################################################################
    # input data
    print(f'Split the data into train/val')
    df = pd.read_csv(args.csv_path)
    df_train, df_val = train_val_split(df)
    print(f"Number of training samples: {len(df_train)}")
    print(f"Number of validation samples: {len(df_val)}")

    # Transforms
    train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(), 
                transforms.Normalize(mean=MEAN, std=STD)])

    val_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(), 
                transforms.Normalize(mean=MEAN, std=STD)])
                

    # training and validation dataset
    train_dataset = DeepFashionDataset(df_train, args.img_dir, transform=train_transform)
    val_dataset = DeepFashionDataset(df_val, args.img_dir, transform=val_transform)
    
    # training and validation data loader
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2*args.num_workers)
    

    ######################################################################
    # Download the model, setup optim & loss fn
    ######################################################################
    # define the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize the model
    print(f'Initialize the model')
    model = MultiHeadResNet18(pretrained=True, requires_grad=False).to(device)

    # Optimizer
    print(f'Initialize the optimizer')
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    
    # loss function
    print(f'Set the loss criterion')
    criterion = loss_fn


    ######################################################################
    # Training
    ######################################################################
    # start the training
    train_loss, val_loss = [], []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1} of {args.epochs}")
        train_epoch_loss = train(model, train_dataloader, optimizer, loss_fn, train_dataset, device)
        val_epoch_loss = validate(model, val_dataloader, loss_fn, val_dataset, device)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}\t Validation Loss: {val_epoch_loss:.4f}")


    ######################################################################
    # Checkpointing
    ######################################################################
    # save the model to disk
    print(f'Save the model to {args.checkpoint_path}')
    save_model(args.epochs, model, optimizer, criterion, args.checkpoint_path)
    
    # save the training and validation loss plot to disk
    print(f'Save the loss graph to {args.lossgraph_path}')
    save_loss_plot(train_loss, val_loss, args.lossgraph_path)


if __name__ == "__main__":
    main()

