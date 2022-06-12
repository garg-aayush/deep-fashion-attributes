# deep-fashion-attributes
This repo contains the pytorch implementation used to train a neural network for predicting the fashion attributes. 

The implementation frames the problem as a multi-label classification case where each label (attribute) is divided into different number of classes. It makes use of pretrained `Resnet18` network trained on `ImageNet` dataset as the base network for transfer learning. The last layer of `Resnet18` is modified to have multiple heads (labels) that are than combined using a crossentropy loss function to train a single network for atttibute predictions.

## Quickstart
#### Setup the environment
```
# clone project
git clone https://https://github.com/garg-aayush/deep-fashion-attributes
cd deep-fashion-attributes

# create conda environment
conda create -n deep_fashion python=3.8
conda activate deep_fashion

# install requirements
pip install -r requirements.txt
```

#### Folder structure
```
pytorch-templates/
├── A_analyze_data.ipynb : Script to analyze and pre-process the input training data
├── B_train.py : Script to train
├── C_test.py : Script to test the trained model on test images
│
├── utils/ : contains helper functions, datasetmodule and models scripts
│   └──...
├── outputs/ : contrains trained model weights
│   └──...
│
├── test_image.jpg : test image
├── test_labeled_image.jpg : predicted test image with labels
├── test_lables_csv.csv : contains the predicted attributes (same format as input train data)
│
└── requirements.txt : file to install python dependencies
```

#### Command line arguments
```
usage: B_train.py [-h] --run_name RUN_NAME [--random_seed RANDOM_SEED] [-ep EPOCHS] [-bs BATCH_SIZE] [-nw NUM_WORKERS] [-lr LEARNING_RATE] [--img_dir IMG_DIR]
                  [--csv_path CSV_PATH] [--checkpoint_path CHECKPOINT_PATH] [--lossgraph_path LOSSGRAPH_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --run_name RUN_NAME
  --random_seed RANDOM_SEED
  -ep EPOCHS, --epochs EPOCHS
                        Total number of training epochs to perform.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for Adam.
  --img_dir IMG_DIR     Path to input images folder
  --csv_path CSV_PATH   Path to input images attributes file
  --checkpoint_path CHECKPOINT_PATH
                        Path to save model
  --lossgraph_path LOSSGRAPH_PATH
                        Path to save loss graph
```

```
usage: C_test.py [-h] -i INPUT [--model MODEL] [--out_image OUT_IMAGE] [--out_csv OUT_CSV]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input image
  --model MODEL         model path
  --out_image OUT_IMAGE
                        output labeled image
  --out_csv OUT_CSV     output labels
```

## Running the scripts
```
# Train with default parameters:
python B_train.py --run_name test

# Traing with learning rate 0.005
python B_train.py --run_name test --learning_rate 0.005

# Testing the trained model
python C_test.py --input test_image.jpg
```

## Observations and further improvements
- The provided dataset consists of images and corresponding three attributes (`neck`, `sleeve_length`, `pattern`) (approx 2200 examples) in the csv file. However, each image_id in attribute file does not has corresponding image in the folder. Therefore, he had to do some pre-processing to the dataset. The jupyter notebook (`A_analyze_data.ipynb`) contains all the steps. 
- The data has a lot of NaN (#N/A) values for each attribute. In order, to fill these values we assign a separate class for NaNs for each attribute.
- Also, the data has a bias towards a particular class for each attribute (see the jupyter notebook). This degrades the model training
- One of the possible ways to handle this bias is to augment the data for attributes and classes that are under-represented. This is something we propose to explore more in future. 
- We currently use pre-trained `ResNet18` network for `ImageNet` for transfer learning. However, the better way could be to use a pretrained classification CNN on [Deep Fashion Dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html). This is something that can also be done to improve the model performance.

## To do
- [X] Analyse the dataset
- [X] Write the dataloader function
- [X] Narrow down on the network and loss function to use
- [X] Write train function and split the data in train/val
- [X] Q.C the results
- [X] Write the test script
- [ ] Create a new branch with code converted to pytorch-lightning scripts with experiment tracker

## Feedback
To give feedback or ask a question or for environment setup issues, you can use the [Github Discussions](https://https://github.com/garg-aayush/deep-fashion-attributes/discussions).
