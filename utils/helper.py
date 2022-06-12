import pandas as pd
import math
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import random

# Set the random seed
def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Spilt the data
def train_val_split(df, train_frac=0.9):
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    num_train_samples = math.floor(len(df) * train_frac)
    num_val_samples = math.floor(len(df) * (1-train_frac))

    train_df = df[:num_train_samples].reset_index(drop=True)
    val_df = df[-num_val_samples:].reset_index(drop=True)

    return train_df, val_df

# custom loss function for multi-label classification
def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    #print(o1.shape)
    #print(t1.shape)
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1 + l2 + l3) / 3

# save the train and validation loss plots to disk
def save_loss_plot(train_loss, val_loss, outpath):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(outpath)
    plt.show()

# save the test image with labels
def save_test_plot(image, cols, labels, outfile=''):
    plt.figure(figsize=(10, 7))
    title = ''
    i = 0
    for col in cols:
        title += f'{col}: {str(labels[i])} '
        i+=1
    plt.imshow(image)
    #print(title)
    plt.title(title)
    if outfile!='':
        plt.savefig(outfile)
    
# save the trained model to disk
def save_model(epochs, model, optimizer, criterion, outpath):
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, outpath)

# Check whether file exists
def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)

# Check whether dir exists
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    