#!/usr/bin/env python
# coding: utf-8

# # Vision Meets ML Assignment 3 : Hands Action Classifier

# ## Import Statements

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


# In[2]:


from engine import train_one_epoch, evaluate
import utils
import transforms as T


# ## Image Augmentation and Transformation 

# In[3]:


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


# In[4]:


class_name_to_index = {'puzzle':0, 'cards':1,'chess':2,'jenga':3}
classes = ('puzzle', 'cards', 'chess', 'jenga')
image_sequence_list_global = []
class_num_to_lst = []
index = 0

directory = "image_sequences"

listOfFileNames = []


# In[5]:


# load images add masks
from torchvision.transforms import functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

'''
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

model = get_instance_segmentation_model(num_classes)

model.to(device)
model.load_state_dict(torch.load('/scratch/sm8235/py_complete_epoch.pth'))
model.eval()
threshold = 0.7

all_images_list = []
count = 0
list.sort(image_sequence_list_global)

for fileName in image_sequence_list_global:
    count += 1
    if(count >= 300):
        break
    currentSequence = fileName
    currentClassName = currentSequence.split('_')[0]
    currentVideoNum = currentSequence.split('_')[1]
    currentSeqNum = currentSequence.split('_')[2]
    currentImgFileName = currentClassName + '_' + currentVideoNum + '_' + currentSeqNum + 'image'

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
    all_images_list.append(img.to('cpu'))
''';


# ## Loading the tensors which were already preprocessed
# 
# In total we have 864 sequence of 10 frames each for all the videos.

# In[6]:


all_images_list = []
for index in range(0,864):
    imgTens = torch.load('combined_images_train' + str(index) + '.pt')
    all_images_list.append(imgTens)
    
print(len(all_images_list))


# In[7]:


array_sequence_list = np.load('combined_image_fileNames_train.npy')
array_sequence_list_test = np.load('combined_image_fileNames_test.npy')

image_sequence_list_global = np.concatenate((array_sequence_list, array_sequence_list_test), axis=None).tolist()
print(len(image_sequence_list_global))


# # Defining the custom dataset for our sequence of images

# In[8]:


# Defining the custom dataset for our sequence of images
from torchvision.transforms import functional as F
class Hand_Seq_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_sequence_list = image_sequence_list_global
    def __getitem__(self, idx):
        # load images and masks
        currentSequence = self.image_sequence_list[idx]
        currentClassName = currentSequence.split('_')[0]
        #img = torch.load('combined_images_train' + str(idx) + '.pt')
        img = all_images_list[idx]
        img = img.to(device)
        labels = torch.ones((10), dtype=torch.uint8)
        labels =  labels.mul(class_name_to_index[currentClassName])
        target = {}
        target["labels"] = labels
        return img, labels

    def __len__(self):
        return len(self.image_sequence_list)


# # Defining the Training Dataloader and Test Dataloader

# In[9]:


# use our dataset and defined transformations
dataset = Hand_Seq_Dataset('',image_transforms['train'])
dataset_test = Hand_Seq_Dataset('',image_transforms['valid'])

# split the dataset in train and test set
#torch.manual_seed(1)
#indices = torch.randperm(len(dataset)).tolist()
dataset_1 = torch.utils.data.Subset(dataset, range(0,576))
dataset_test_1 = torch.utils.data.Subset(dataset_test, range(576,864))

print(len(dataset_1))
print(len(dataset_test_1))

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset_1, batch_size=4, shuffle=True, num_workers=0,collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(dataset_test_1, batch_size=4, shuffle=False, num_workers=0,collate_fn=utils.collate_fn)
classes = ('puzzle', 'cards', 'chess', 'jenga')


# # Defining the model and adding layers required for hand action classifier

# In[11]:


import torchvision.models as models
import torch.nn as nn


# In[12]:


class Net_Video_Classification(nn.Module):
    def __init__(self, sinkhorn_iter=0):
        super().__init__()
        self.res50_model = models.resnet50(pretrained=True)
        for param in self.res50_model.parameters():
            param.requires_grad = False
        self.layers =[nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)]
        self.layers.extend(list(self.res50_model.children())[1:-1])
        self.res50_conv = nn.Sequential(*self.layers)
        self.pretrained_resnet = self.res50_conv
        self.fc1 = nn.Linear(2048, 400)
        self.fc2 = nn.Linear(400, 4)
        self.lr = nn.LeakyReLU(0.2,True)
        self.sm = nn.Softmax()
        
    def forward(self, x):
        # Split input into four pieces and pass them into the
        # same convolutional neural network.
        pre_trained_last = self.pretrained_resnet(x)
        pre_trained_last_flatten = pre_trained_last.view(10,-1)
        #print(pre_trained_last_flatten)
        #pre_trained_last_final = self.conv1(pre_trained_last_flatten)
        pre_trained_last_1 = self.lr(self.fc1(pre_trained_last_flatten))
        pre_trained_last_final = self.lr(self.fc2(pre_trained_last_1))
        return self.sm(pre_trained_last_final)


# ### Below we can see : conv0, fc1, fc2 layers were added.

# In[13]:


net = Net_Video_Classification()
for param in net.named_parameters():
    if(param[1].requires_grad == True):
        print(param[0])
net = net.cuda()


# ## Script for each training step

# In[ ]:


def train_step(inputs, labels, optimizer, criterion, unet):
    optimizer.zero_grad()
    outputs = unet(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


# ## Set the optimizer and the criterion

# In[ ]:


from torch import optim
learning_rate  = 0.001
network_momentum = 0.99

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate, momentum = network_momentum)
optimizer = optim.Adam(net.parameters())


# # Train the model for 20 epochs and save the .pth files

# In[ ]:


from tqdm import trange
from matplotlib import pyplot as plt
epochs = 10
t = trange(epochs, leave=True)
net.train()

for iter in t:
    total_loss = 0
    for index, data in enumerate(data_loader):
        img,labels = data
        labels = labels[0].long().to(device)
        img = image_transforms['train'](img)
        inp=img[0].to(device)

        with torch.autograd.detect_anomaly():        
            batch_loss = train_step(inp, labels, optimizer, criterion, net)
            total_loss += batch_loss
    print("\n\n\n******** total_epoch_training_loss = " + str(total_loss/len(data_loader))+" ********\n\n\n")
    PATH = '/scratch/sm8235/resnet_saved/1saved_' + str(iter) + '_l_' + str(total_loss/len(data_loader)) +  '.pth'
    torch.save(net.state_dict(),PATH)
PATH = '/scratch/sm8235/resnet_saved/1final_10epoch.pth'
torch.save(net.state_dict(),PATH)


# ## Defining the test step

# In[ ]:


def test_step(inputs, labels, optimizer, criterion, unet):
    optimizer.zero_grad()
    outputs = unet(inputs)
    loss = criterion(outputs, labels)
    print(loss)
    return outputs


# In[ ]:


total_loss = 0
for index, data in enumerate(data_loader_test):
    img,labels = data
    labels = labels[0].long().to(device)
    inp=img[0].to(device)
    with torch.autograd.detect_anomaly():        
        outputs_test = test_step(inp , labels, optimizer, criterion, net)


# # Accuracy of ResNet50 model on the test data set

# In[ ]:


correct = 0
total = 0
with torch.no_grad():
    for data in data_loader_test:
        images, labels = data
        outputs = net(images[0])
        _, predicted = torch.max(outputs.data, 1)
        total += labels[0].size(0)
        label = labels[0].to(device)
        predicted = predicted.to(device)
        correct += (predicted == label).sum().item()

print('Accuracy of the network on all 16 Test Set videos: %d %%' % (100 * correct / total))


# # Model 2 : vgg16 net

# In[14]:


from torchvision import models
from torch import optim
model = models.vgg16(pretrained=True)

# Freeze model weights
for param in model.parameters():
    param.requires_grad = False
n_classes = 4
import torch.nn as nn
# Add on classifier
model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, n_classes),                   
                      nn.LogSoftmax(dim=1))
model = model.to('cuda')
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# # Code for training the model for 20 epochs

# In[ ]:


from tqdm import trange
n_epochs = 20
total_loss = 0
t = trange(n_epochs, leave=True)
for epoch in t:
    total_loss = 0
    for data, targets in data_loader:
        # Generate predictions
        targets = targets[0].long().to(device)
        data=data[0].to(device)
        out = model(data[:,:3])
        # Calculate loss
        optimizer.zero_grad()
        loss = criterion(out, targets)
        # Backpropagation
        loss.backward()
        # Update model parameters
        optimizer.step()      
        total_loss += loss
    print("\n\n\n******** total_epoch_training_loss = " + str(total_loss/len(data_loader))+" ********\n\n\n")
    PATH = '/scratch/sm8235/vggnet_saved/saved_' + str(iter) + '_l_' + str(total_loss/len(data_loader)) +  '.pth'
    torch.save(net.state_dict(),PATH)
PATH = '/scratch/sm8235/vggnet_saved/final_20epoch.pth'
torch.save(net.state_dict(),PATH)


# # Checking some of the outputs

# In[39]:


for data, targets in data_loader_test:
    data=data[0].to(device)
    log_ps = model(data[:,:3])
    # Convert to probabilities
    ps = torch.exp(log_ps)
        # Find predictions and correct
    pred = torch.max(ps, dim=1)
    print(pred)
    targets = targets[0].to(device)
    print(targets)


# # Accuracy of vggnet16 model on the test data set

# In[41]:


correct = 0
total = 0
with torch.no_grad():
    for data in data_loader_test:
        images, labels = data
        outputs = model(images[0][:,:3])
        _, predicted = torch.max(outputs.data, 1)
        total += labels[0].size(0)
        label = labels[0].to(device)
        predicted = predicted.to(device)
        correct += (predicted == label).sum().item()

print('Accuracy of the network on all 16 Test Set videos: %d %%' % (100 * correct / total))


# In[ ]:




