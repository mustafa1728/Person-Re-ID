# Evaluation 
# Acknowledgement: the code is based on Siddhant Kapil's repo on LA-Transformer

from __future__ import print_function

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
from baselines.AlignedReID.model import AlignedReIDModel
from baselines.Centroids_reid.model import CentroidReID
from baselines.Centroids_cam_reid.model import CentroidCamReID
from baselines.TransReID.model import TransReID
import logging
logging.basicConfig(filename="experiments.log", filemode='a', format='%(levelname)s | %(message)s', level=logging.INFO)


batch_size = 1

# ### Load Model
#save_path = "<model weight path>"
#model = ReidModel(num_classes=C)
#model.load_state_dict(torch.load(save_path), strict=False)
#model.eval()

# TODO: Comment out the dummy model
######## LA_Transfoermer
# H, W, D = 1, 14, 768
# vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
# model = LATransformerTest(vit_base, lmbd=8).to("cpu")
# save_path = os.path.join('./baselines/LA_Transformer/net_best.pth')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()
# logging.info("LA Transformer")

# ######## Aligned ReID
# H, W, D = 1, 1, 2048
# model = AlignedReIDModel()
# save_path = os.path.join('baselines/AlignedReID/la_tr_checkpoint_ep120.pth.tar')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()
# logging.info("Aligned ReID")

# ######## Centroid ReID
# H, W, D = 1, 1, 2048
# model = CentroidReID()
# save_path = os.path.join('baselines/Centroids_reid/epoch=29.ckpt')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()
# logging.info("Centroid ReID")

######## Centroid ReID with cam embeddings
# H, W, D = 1, 1, 2048
# model = CentroidCamReID()
# save_path = os.path.join('baselines/Centroids_cam_reid/epoch=29.ckpt')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()
# logging.info("Centroid ReID with cam embeddings")

######## TransReID with cam embeddings
H, W, D = 1, 197, 768
model = TransReID()
save_path = os.path.join('baselines/TransReID/tranreid_120.pth')
model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
model.eval()
logging.info("TransReID")



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


image_datasets = {}
data_dir = "data/val"

image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'),
                                          data_transforms['query'])
image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'),
                                          data_transforms['gallery'])
query_loader = DataLoader(dataset = image_datasets['query'], batch_size=batch_size, shuffle=False )
gallery_loader = DataLoader(dataset = image_datasets['gallery'], batch_size=batch_size, shuffle=False)

class_names = image_datasets['query'].classes


# ###  Extract Features

def extract_feature(dataloaders):
    
    features =  torch.FloatTensor()
    count = 0
    idx = 0
    for data in tqdm(dataloaders):
        img, label = data
        # Uncomment if using GPU for inference
        #img, label = img.cuda(), label.cuda()

        output = model(img) # (B, D, H, W) --> B: batch size, HxWxD: feature volume size
        # print(output.size())

        n, c, h, w = img.size()
        
        count += n
        features = torch.cat((features, output.detach().cpu()), 0)
        idx += 1
    return features

# Extract Query Features

query_feature= extract_feature(query_loader)

# Extract Gallery Features

gallery_feature = extract_feature(gallery_loader)

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

# index = faiss.IndexIDMap(faiss.IndexFlatIP(H*W*D))
index = faiss.IndexIDMap(faiss.IndexFlatL2(H*W*D))

index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]),np.array(gallery_label))

def search(query: str, k=1):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    return top_k


# ### Evaluate 

rank1_score = 0
rank5_score = 0
ap = 0
count = 0
for query, label in zip(concatenated_query_vectors, query_label):
    count += 1
    label = label
    output = search(query, k=10)
    rank1_score += rank1(label, output) 
    rank5_score += rank5(label, output) 
    print("Correct: {}, Total: {}, Incorrect: {}".format(rank1_score, count, count-rank1_score), end="\r")
    ap += calc_ap(label, output)

str_to_print = "Rank1: %.3f, Rank5: %.3f, mAP: %.3f"%(rank1_score/len(query_feature), 
                                             rank5_score/len(query_feature), 
                                             ap/len(query_feature))
print(str_to_print)
logging.info(str_to_print)
                                             
                                                 

