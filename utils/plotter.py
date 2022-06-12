import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_images(images, labels, preds, name_classes, nimg=12, ncols=3,
                data_mean=[], data_std=[]):
    nrows = nimg//ncols  
    # define figure
    fig_, axes=plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.ravel()

    #print(np.min(images), np.max(images))
    for i in range(nimg):
        label_name = name_classes[labels[i]]
        pred_name = name_classes[preds[i]]
        image = images[i]
        image[0] = image[0]*data_std[0] + data_mean[0]
        image[1] = image[1]*data_std[1] + data_mean[1]
        image[2] = image[1]*data_std[2] + data_mean[2]
        #print(np.min(image), np.max(image))
        image = np.transpose((image*255).astype('uint8'), (1,2,0))
        axes[i].imshow(image)
        axes[i].set_title(f'label: {label_name} \n pred: {pred_name}', fontsize=8)
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.2)
    return fig_ 

def plot_examples(df, img_dir, nimgs, idx, col_names, ncols=3):
    nrows = nimgs//ncols  
    # define figure
    fig_, axes=plt.subplots(nrows, ncols, figsize=(12, 16))
    axes = axes.ravel()
    j = 0

    for i in idx:
        img_path = img_dir + df.filename[i]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[j].imshow(image)
        title = ''
        for col in col_names:
            title = title + f' {col}: {df[col][0]}'
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.2)
    return fig_ 