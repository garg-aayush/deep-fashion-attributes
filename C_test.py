import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import argparse
from utils.models import MultiHeadResNet18
from utils.helper import save_test_plot, file_path
import csv

# IMAGESIZE (P x P)
IMAGE_SIZE = 224

# IMAGENET mean
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ATTRIBUTE NAMES
COL_NAMES= ['filename', 'neck', 'sleeve_length', 'pattern']

######################################################################
# Main function
######################################################################
def main():
    ######################################################################
    # input cmdline arguments
    ######################################################################
    parser=argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('-i', '--input', required=True, help='path to input image')
    parser.add_argument("--model", default='./outputs/model.pth', type=file_path, help="model path")
    parser.add_argument("--out_image", default='./test_labeled_image.jpg', help="output labeled image")
    parser.add_argument("--out_csv", default='./test_labels_csv.csv', help="output labels")
    args=parser.parse_args()

    ######################################################################
    # define the computation device
    ######################################################################
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')
    # model
    model = MultiHeadResNet18(pretrained=False, requires_grad=False)
    print(f'Load trained model from {args.model}')
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)])
        

    # read an image
    image = cv2.imread(args.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_image = image.copy()
    
    # apply image transforms
    image = transform(image)
    # add batch dimension
    image = image.unsqueeze(0).to(device)

    # forward pass
    outputs = model(image)
    
    # extract the three output
    output1, output2, output3 = outputs
    
    # get the index positions of the highest label score
    out_label_1 = np.argmax(output1.detach().cpu())
    out_label_2 = np.argmax(output2.detach().cpu())
    out_label_3 = np.argmax(output3.detach().cpu())

    # convert last label value (if any) to #N/A
    if out_label_1 == 7:
        out_label_1 = '#N/A'
    if out_label_2 == 4:
        out_label_2 = '#N/A'
    if out_label_3 == 10:
        out_label_3 = '#N/A'

    # tensor to int
    out_labels = [out_label_1.item(), out_label_2.item(), out_label_3.item()]
    
    # save output image
    print(f'Write output labeled image: {args.out_image}')
    save_test_plot(orig_image, COL_NAMES[1:], out_labels, outfile=args.out_image)
    
    # save attribute file
    out_value = [args.input] + out_labels
    print(f'Write output labels csv: {args.out_csv}')
    
    with open(args.out_csv, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(COL_NAMES)
        write.writerow(out_value)

if __name__=='__main__':
    main()