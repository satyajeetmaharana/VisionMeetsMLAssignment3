#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from io import StringIO
import csv
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# In[2]:


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
 
num_classes = 2  # 1 class (hand) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 


# In[3]:


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


# In[4]:


from engine import train_one_epoch, evaluate
import utils
import transforms as T

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# In[5]:


from torchvision import transforms

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[6]:


model = get_instance_segmentation_model(num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.load_state_dict(torch.load('/scratch/sm8235/py_complete_epoch.pth'))
model.eval()
print(device)


# In[7]:


class_name_to_index = {'puzzle':0, 'cards':1,'chess':2,'jenga':3}
classes = ('puzzle', 'cards', 'chess', 'jenga')
image_sequence_list_global = []
class_num_to_lst = []
index = 0


training_for_each_class = {'puzzle':0, 'cards':0, 'chess':0,'jenga':0}
directory = "image_sequences"

listOfFileNames = []
image_sequence_list_global = []
image_sequence_list_final = []

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        videoName = os.path.join(filename)
        className = videoName.split('_')[0]
        videoNum = videoName.split('_')[1]
        class_num_to_lst.append(className + '_' + videoNum)
class_num_to_set = set(class_num_to_lst)

for videoName in class_num_to_set:
    className = videoName.split('_')[0]
    videoNum = videoName.split('_')[1]
    for seqNum in range(0,18):
        image_sequence_list_global.append(className + '_' + videoNum + '_' + str(seqNum))


# In[ ]:


# load images add masks
from torchvision.transforms import functional as F

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

model = get_instance_segmentation_model(num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.load_state_dict(torch.load('/scratch/sm8235/py_complete_epoch.pth'))
model.eval()
threshold = 0.7

all_images_list = []

list.sort(image_sequence_list_global)

training_for_each_class = {'puzzle':0, 'cards':0, 'chess':0,'jenga':0}
set_videos_done = set()

for fileName in image_sequence_list_global:
    currentSequence = fileName
    currentClassName = currentSequence.split('_')[0]
    currentVideoNum = currentSequence.split('_')[1]
    currentSeqNum = currentSequence.split('_')[2]
    currentImgFileName = currentClassName + '_' + currentVideoNum + '_' + currentSeqNum + 'image'
    if(training_for_each_class[currentClassName] >= 90):
        continue
    training_for_each_class[currentClassName] += 1
    set_videos_done.add(currentVideoNum)
    all_images_combined = []

    for imgNum in range(0,10):
        img_path = os.path.join('', "image_sequences",currentImgFileName + str(imgNum) + '.jpg')
        transforms = get_transform(train=True)
        target = {}
        img = Image.open(img_path).convert("RGB")
        with torch.autograd.detect_anomaly():
            img = image_transforms['train'](img)
            #img = F.to_tensor(img)
        with torch.no_grad():
            prediction = model([torch.tensor(img).to(device)])
            maskImage = torch.sum(prediction[0]['masks'][:, 0],dim=0)
            outMaskImage = (maskImage>threshold).float()
            outMaskImage = outMaskImage.expand(1,-1,-1)
            # outMaskImage : H, W
            combinedImage = torch.cat([torch.tensor(img).to(device),outMaskImage],dim=0)
            combinedImage = combinedImage.expand(1,-1,-1,-1)
            all_images_combined.append(combinedImage)
            #print(combinedImage.shape)
    img = torch.cat(all_images_combined,dim=0)
    image_sequence_list_final.append(fileName)
    all_images_list.append(img.to('cpu'))


# In[ ]:


np.save('combined_images_train.npy',np.array([t.numpy() for t in all_images_list]))
np.save('combined_image_fileNames_train.npy',np.array(image_sequence_list_final))


# In[ ]:


from torchvision.transforms import functional as F

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

model = get_instance_segmentation_model(num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.load_state_dict(torch.load('/scratch/sm8235/py_complete_epoch.pth'))
model.eval()
threshold = 0.7
all_images_list = []
image_sequence_list_final = []

list.sort(image_sequence_list_global)
training_for_each_class = {'puzzle':0, 'cards':0, 'chess':0,'jenga':0}
set_videos_done = set()

count = 0

for fileName in image_sequence_list_global:
    currentSequence = fileName
    currentClassName = currentSequence.split('_')[0]
    currentVideoNum = currentSequence.split('_')[1]
    currentSeqNum = currentSequence.split('_')[2]
    currentImgFileName = currentClassName + '_' + currentVideoNum + '_' + currentSeqNum + 'image'
    if(training_for_each_class[currentClassName] >= 8):
        continue
    training_for_each_class[currentClassName] += 1
    set_videos_done.add(currentVideoNum)



for fileName in image_sequence_list_global:
    currentSequence = fileName
    currentClassName = currentSequence.split('_')[0]
    currentVideoNum = currentSequence.split('_')[1]
    currentSeqNum = currentSequence.split('_')[2]
    currentImgFileName = currentClassName + '_' + currentVideoNum + '_' + currentSeqNum + 'image'
    if(currentVideoNum in set_videos_done):
        continue
    all_images_combined = []
    count += 1
    if(count > 36):
        break

    for imgNum in range(0,10):
        img_path = os.path.join('', "image_sequences",currentImgFileName + str(imgNum) + '.jpg')
        target = {}
        img = Image.open(img_path).convert("RGB")
        with torch.autograd.detect_anomaly():
            img = image_transforms['train'](img)
            #img = F.to_tensor(img)
        with torch.no_grad():
            prediction = model([torch.tensor(img).to(device)])
            maskImage = torch.sum(prediction[0]['masks'][:, 0],dim=0)
            outMaskImage = (maskImage>threshold).float()
            outMaskImage = outMaskImage.expand(1,-1,-1)
            # outMaskImage : H, W
            combinedImage = torch.cat([torch.tensor(img).to(device),outMaskImage],dim=0)
            combinedImage = combinedImage.expand(1,-1,-1,-1)
            all_images_combined.append(combinedImage)
            #print(combinedImage.shape)
    img = torch.cat(all_images_combined,dim=0)
    image_sequence_list_final.append(fileName)
    all_images_list.append(img.to('cpu'))


# In[ ]:


np.save('combined_images_test.npy',np.array([t.numpy() for t in all_images_list]))
np.save('combined_image_fileNames_test.npy',np.array(image_sequence_list_final))


# In[ ]:





# In[ ]:





# In[ ]:




