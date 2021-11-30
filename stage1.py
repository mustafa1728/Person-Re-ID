import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn as MaskRCNN
from torchvision.models.segmentation import fcn_resnet50
import cv2
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

import argparse

USE_GPU = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Custom HoG')
parser.add_argument('--root', type=str, help='path to dataset root directory', default="/Users/mustafa/Desktop/IIT Delhi/acads/COL780/assignments/project/reid-col780-master/data/", required=False)
args = parser.parse_args()

def get_prediction(img_path, model, threshold):
    img = Image.open(img_path)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img = transform(img)
    pred = model([img])
    masks = (pred[0]['masks']).squeeze().detach().cpu().numpy()
    masks = masks
    return masks

def get_prediction_segm(img_path, model, threshold):
    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    img = Image.open(img_path)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img = transform(img)
    pred = model(img.view(([1] + list(img.size()))))['out']
    normalized_masks = torch.nn.functional.softmax(pred, dim=1)
    person_mask = None
    for i in range(normalized_masks.size(1)):
        if i == sem_class_to_idx['person']:
            if person_mask is None:
                person_mask = normalized_masks[0, i]
            else:
                person_mask += normalized_masks[0, i]

    # print(person_mask.min(), person_mask.max(), person_mask.size())
    person_mask = (person_mask - person_mask.min()) / (person_mask.max() - person_mask.min())
    person_mask = person_mask > (person_mask.min() + person_mask.max())/3
    return person_mask.detach().numpy()

  

def main():
    # model = MaskRCNN(
    #     pretrained=True, 
    #     progress=True, 
    #     pretrained_backbone=True 
    # )
    model = fcn_resnet50(pretrained=True, progress=False)
    if USE_GPU:
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()

    root_dir = args.root

    for subdir, dirs, files in os.walk(root_dir):
        if ".DS_Store" in subdir:
            continue
        print(os.path.basename(subdir))
        for file in files:
            if ".DS_Store" in file:
                continue
            img_path = os.path.join(subdir, file)
            save_path = img_path.replace("data", "masks")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # masks = get_prediction(img_path, model, 0.5)
            # mask = masks[0]
            mask = get_prediction_segm(img_path, model, 0.5)
            mask = np.uint(255*mask)

            # img = cv2.imread(img_path)
            # masked_img = img * np.expand_dims(mask, axis=-1)
            cv2.imwrite(save_path, mask)


    


main()