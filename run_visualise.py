# Evaluation 
# Acknowledgement: the code is based on Siddhant Kapil's repo on LA-Transformer

from __future__ import print_function

import cv2
import os
import faiss
import numpy as np

from PIL import Image
from tqdm import tqdm
import timm

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from model import ReidModel, DummyModel
from utils import get_id
from metrics import rank1, rank5, calc_ap

from baselines.LA_Transformer.model import LATransformerTest
from baselines.AlignedReID.model import AlignedReIDModel, MaskAlignedReIDModel, AlignedReIDCam
from baselines.Centroids_reid.model import CentroidReID
from baselines.Centroids_cam_reid.model import CentroidCamReID
# from baselines.TransReID.model import TransReID
import logging
logging.basicConfig(filename="experiments.log", filemode='a', format='%(levelname)s | %(message)s', level=logging.INFO)

batch_size = 1

save_dir_root = "visualisation"
os.makedirs(save_dir_root,  exist_ok=True)


# TODO: Comment out the dummy model
######## LA_Transformer Baseline
# H, W, D = 1, 14, 768
# name = "LATransformer_baseline"
# vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
# model = LATransformerTest(vit_base, lmbd=8).to("cpu")
# save_path = os.path.join('./weights/La_Transformer_baseline.pth')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()

######## LA_Transformer Improved
# H, W, D = 1, 14, 768
# name = "LATransformer_improved"
# vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
# model = LATransformerTest(vit_base, lmbd=8).to("cpu")
# save_path = os.path.join('./weights/La_Transformer_Triplet_Self_Ensemble.pth')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()

# ######## Aligned ReID
H, W, D = 1, 1, 2048
name = "AlignedReID"
model = AlignedReIDModel()
save_path = os.path.join('./weights/AlignedReID_baseline.pth.tar')
model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
model.eval()


# ######## Aligned ReID Improved
# H, W, D = 1, 1, 2048
# name = "AlignedReID_improved"
# model = AlignedReIDModel()
# save_path = os.path.join('./weights/AlignedReID_Feature_Invariance.pth.tar')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
# model.eval()


# ######## Mask Guided Aligned ReID
# H, W, D = 1, 1, 2048
# name = "Mask_Guided_AlignedReID"
# model = MaskAlignedReIDModel()
# save_path = os.path.join('./weights/AlignedReID_baseline.pth.tar')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
# model.eval()

# ######## Camera Aligned ReID
# H, W, D = 1, 1, 2048
# name = "Camera_AlignedReID"
# model = AlignedReIDCam()
# save_path = os.path.join('baselines/AlignedReID/checkpoint_ep120.pth.tar')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
# model.eval()

# ######## Centroid ReID
# H, W, D = 1, 1, 2048
# name = "CentroidReID"
# model = CentroidReID()
# save_path = os.path.join('baselines/Centroids_reid/epoch=29.ckpt')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()


######## Centroid ReID with cam embeddings
# H, W, D = 1, 1, 2048
# name = "CentroidCamReID"
# model = CentroidCamReID()
# save_path = os.path.join('baselines/Centroids_cam_reid/epoch=29.ckpt')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()

######## TransReID 
# H, W, D = 1, 197, 768
# name = "TransReID"
# model = TransReID()
# save_path = os.path.join('baselines/TransReID/tranreid_120.pth')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()



# ### Data Loader for query and gallery

# TODO: For demo, we have resized to 224x224 during data augmentation
# You are free to use augmentations of your own choice
transform_query_list = [
        transforms.Resize((224,224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_gallery_list = [
        transforms.Resize(size=(224,224), interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

data_transforms = {
        'query': transforms.Compose( transform_query_list ),
        'gallery': transforms.Compose(transform_gallery_list),
    }

transform_raw_query_list = [
        transforms.ToTensor(),
    ]
transform_raw_gallery_list = [
        transforms.ToTensor(),
    ]

data_transforms_raw = {
        'query': transforms.Compose( transform_raw_query_list ),
        'gallery': transforms.Compose(transform_raw_gallery_list),
    }


image_datasets = {}
data_dir = "data/val"

image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'),
                                          data_transforms['query'])
image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'),
                                          data_transforms['gallery'])
query_loader = DataLoader(dataset = image_datasets['query'], batch_size=batch_size, shuffle=False )
gallery_loader = DataLoader(dataset = image_datasets['gallery'], batch_size=batch_size, shuffle=False)
image_datasets['query_raw'] = datasets.ImageFolder(os.path.join(data_dir, 'query'),
                                          data_transforms_raw['query'])
image_datasets['gallery_raw'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'),
                                          data_transforms_raw['gallery'])
query_raw_loader = DataLoader(dataset = image_datasets['query_raw'], batch_size=batch_size, shuffle=False )
gallery_raw_loader = DataLoader(dataset = image_datasets['gallery_raw'], batch_size=batch_size, shuffle=False)


class_names = image_datasets['query'].classes


# ###  Extract Features

def extract_feature(dataloaders, raw_dataloader):
    
    features =  torch.FloatTensor()
    images =  torch.FloatTensor()
    # images =  []
    count = 0
    idx = 0
    for data, raw_data in tqdm(zip(dataloaders, raw_dataloader)):
        img, label = data
        raw_img, _ = raw_data
        # Uncomment if using GPU for inference
        #img, label = img.cuda(), label.cuda()

        output = model(img) # (B, D, H, W) --> B: batch size, HxWxD: feature volume size
        # print(output.size())

        n, c, h, w = img.size()
        
        count += n
        features = torch.cat((features, output.detach().cpu()), 0)
        images = torch.cat((images, raw_img.detach().cpu()), 0)
        idx += 1
    return features, images

# Extract Query Features

query_feature, query_imgs= extract_feature(query_loader, query_raw_loader)

# Extract Gallery Features

gallery_feature, gallery_imgs = extract_feature(gallery_loader, gallery_raw_loader)

# Retrieve labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)


# ## Concat Averaged GELTs

concatenated_query_vectors = []
for query in tqdm(query_feature):
    fnorm = torch.norm(query, p=2, dim=1, keepdim=True)#*np.sqrt(H*W)
    query_norm = query.div(fnorm.expand_as(query))
    concatenated_query_vectors.append(query_norm.view((-1)))

concatenated_gallery_vectors = []
for gallery in tqdm(gallery_feature):
    fnorm = torch.norm(gallery, p=2, dim=1, keepdim=True)#*np.sqrt(H*W)
    gallery_norm = gallery.div(fnorm.expand_as(gallery))
    concatenated_gallery_vectors.append(gallery_norm.view((-1)))
  

# ## Calculate Similarity using FAISS

index = faiss.IndexIDMap(faiss.IndexFlatIP(H*W*D))
# index = faiss.IndexIDMap(faiss.IndexFlatL2(H*W*D))
ids = [i for i in range(len(gallery_imgs))]
index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]),np.array(ids))

label_index = faiss.IndexIDMap(faiss.IndexFlatIP(H*W*D))
label_index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]), np.array(gallery_label))

def search_imgs(query: str, k=1):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    return top_k

def search(query: str, k=1):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = label_index.search(encoded_query, k)
    return top_k


# ### Evaluate 
save_dir = os.path.join(save_dir_root, name)
os.makedirs(save_dir, exist_ok=True)
rank1_score = 0
rank5_score = 0
ap = 0
count = 0
for query, query_img, label in zip(concatenated_query_vectors, query_imgs, query_label):
    count += 1
    label = label
    ids = search_imgs(query, k=10)
    # print(ids[1][0])
    ids = ids[1][0] 
    images = gallery_imgs[ids]

    query_save_dir = os.path.join(save_dir, str(count))
    os.makedirs(query_save_dir, exist_ok=True)

    query_img_to_save = np.uint8(np.array(255*query_img))
    # print(query_img_to_save.shape)
    query_img_to_save = np.transpose(query_img_to_save, (1, 2, 0))
    query_img_to_save = cv2.cvtColor(query_img_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(query_save_dir, "query.png"), query_img_to_save)
    for id, img in enumerate(images):
        img_to_save = np.uint8(np.array(255*img))
        img_to_save = np.transpose(img_to_save, (1, 2, 0))
        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(query_save_dir, "top_{}.png".format(id+1)), img_to_save)
    

    output = search(query, k=10)
    rank1_score += rank1(label, output) 
    rank5_score += rank5(label, output) 
    print("Correct: {}, Total: {}, Incorrect: {}".format(rank1_score, count, count-rank1_score), end="\r")
    ap += calc_ap(label, output)

str_to_print = "Rank1: %.3f, Rank5: %.3f, mAP: %.3f"%(rank1_score/len(query_feature), 
                                             rank5_score/len(query_feature), 
                                             ap/len(query_feature))
print("")
print(name)
print(str_to_print)
logging.info(name)
logging.info(str_to_print)