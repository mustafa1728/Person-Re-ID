from __future__ import print_function

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# Evaluation 
# Acknowledgement: the code is based on Siddhant Kapil's repo on LA-Transformer

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
from baselines.AlignedReID.model import AlignedReIDModel, MaskAlignedReIDModel
from baselines.Centroids_reid.model import CentroidReID
from baselines.Centroids_cam_reid.model import CentroidCamReID
# from baselines.TransReID.model import TransReID
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
# # save_path = os.path.join('./baselines/LA_Transformer/net_best.pth')
# save_path = os.path.join('./baselines/LA_Transformer/ema_triplet_net_best.pth')
# # save_path = os.path.join('./baselines/LA_Transformer/model_30.pth')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()
# logging.info("LA Transformer with EMA Combined")

# ######## Aligned ReID
H, W, D = 1, 1, 2048
model = AlignedReIDModel()
save_path = os.path.join('baselines/AlignedReID/checkpoint_ep120.pth.tar')
# model = MaskAlignedReIDModel()
# save_path = os.path.join('baselines/AlignedReID/masked_checkpoint_ep160.pth.tar')
model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
# model = torch.load(save_path, map_location=torch.device('cpu'))
# print(model.keys())
model.eval()
# # logging.info("Aligned ReID on soft masks total test only")
# logging.info("Aligned ReID with mask guidance on masked data")

# ######## Centroid ReID
# H, W, D = 1, 1, 2048
# model = CentroidReID()
# save_path = os.path.join('baselines/Centroids_reid/epoch=29.ckpt')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()
# logging.info("Centroid ReID")
# logging.info("Centroid ReID on soft masks total test only")

######## Centroid ReID with cam embeddings
# H, W, D = 1, 1, 2048
# model = CentroidCamReID()
# save_path = os.path.join('baselines/Centroids_cam_reid/epoch=29.ckpt')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()
# logging.info("Centroid ReID with cam embeddings")

######## TransReID with cam embeddings
# H, W, D = 1, 197, 768
# model = TransReID()
# save_path = os.path.join('baselines/TransReID/tranreid_120.pth')
# model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')), strict=False)
# model.eval()
# logging.info("TransReID")



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
# data_dir = "masked_data/val"

image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'),
                                          data_transforms['query'])
image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'),
                                          data_transforms['gallery'])
query_loader = DataLoader(dataset = image_datasets['query'], batch_size=batch_size, shuffle=False )
gallery_loader = DataLoader(dataset = image_datasets['gallery'], batch_size=batch_size, shuffle=False)

class_names = image_datasets['query'].classes


# ###  Extract Features

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

cmap = get_cmap(12)

colors = [(int(np.random.choice(range(256))), int(np.random.choice(range(256))), int(np.random.choice(range(256)))) for i in range(12)]
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

gallery_path = image_datasets['gallery'].imgs
gallery_cam,gallery_label = get_id(gallery_path)

def plot_tsne(dataloaders, save_path = "tsne.png", title = "tsne"):
    
    features =  torch.FloatTensor()
    vectors =  torch.FloatTensor()
    count = 0
    idx = 0
    labels = []


    for data in tqdm(dataloaders):
        img, label = data
        # Uncomment if using GPU for inference
        #img, label = img.cuda(), label.cuda()

        output = model(img) # (B, D, H, W) --> B: batch size, HxWxD: feature volume size
        # print(output.size())
        vector = output.view(1, -1)
        labels.append(label.cpu().numpy())



        n, c, h, w = img.size()
        
        count += n
        # features = torch.cat((features, output.detach().cpu()), 0)
        vectors = torch.cat((vectors, vector.detach().cpu()), 0)
        idx += 1
        # if idx > 2:
        #     break

    # labels = np.asarray(labels).reshape(-1).tolist()
    # print(labels)
    set_labels = list(set(gallery_label))
    label_to_id = {l:i for i, l in enumerate(set_labels)}

    print(vectors.size())
    red_vectors = tsne.fit_transform(vectors)
    print(red_vectors.shape)

    for i in range(red_vectors.shape[0]):
        x, y = red_vectors[i]
        plt.scatter([x], [y], c=cmap(label_to_id[gallery_label[i]]), s=5)
    plt.title(title)
    plt.savefig(save_path, dpi=300)
    


plot_tsne(gallery_loader, save_path = "./visualisation/AlignedReID_baseline.png", title="TSNE plot for AlignedReID")
# plot_tsne(gallery_loader, save_path = "./visualisation/LA_Transformer_improved.png", title="TSNE plot for LA Transformer")
# 
