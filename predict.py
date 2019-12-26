#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa


# In[20]:


# import some libraries you maybe use
import torchvision # an useful library to help I/O (highly recommend). To install this, just do "pip install torchvision"
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import visdom
import scipy
from utils.plot_cm import plot_cm


# In[3]:


batch_size = 100


# In[4]:


ROOT_DIR = './models'
TRAINING_NAME = 'phase'
OUTPUT_DIR = os.path.join(ROOT_DIR, TRAINING_NAME)
MODEL_PATH = os.path.join(OUTPUT_DIR, 'weight.pth')
#ACC_MODEL_PATH = os.path.join(OUTPUT_DIR, 'weight_acc.pth')
#RESULT_PATH = os.path.join(OUTPUT_DIR, 'result.csv')
#LOG_PATH = os.path.join(OUTPUT_DIR, 'log')
#FIG_PATH = os.path.join(OUTPUT_DIR, 'cm.png')





# # Data loading and preprocessing
# In order to train the model with training data, the first step is to read the data from your folder, database, etc. The below is just an example.

# In[6]:


from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize, Normalize
from torch.utils.data import DataLoader
import os
# Define path to your dataset
dataset = "./data" # the root folder
trainpath = os.path.join(dataset,"train") # train set
valpath = os.path.join(dataset,"val") # validation set
def pad(raw):
    ex = np.zeros((22050,))
    ex[:len(raw)] = raw
    return ex

norm =  lambda x: (x.astype(np.float32) / (np.max(x)+1e-6))*0.5
spct = lambda x: scipy.signal.spectrogram(x ,fs= 10e3,mode='phase')[2] #overlap
tri = lambda x: [x, x, x]
totensor = lambda x: torch.Tensor(x)

tsfm = Compose([
        pad, # rescale to -1 to 1
        norm, # rescale to -1 to 1
        spct, # MFCC 
        tri,
        totensor
        ])

nploader = np.load


# In[7]:


traindata = DatasetFolder(root=trainpath, loader=nploader, transform=tsfm, extensions=['npy'])
valdata = DatasetFolder(root=valpath, loader=nploader, transform=tsfm, extensions=['npy'])

# Create a loader
trainloader = DataLoader(traindata,batch_size=batch_size,shuffle=True, pin_memory=True, num_workers=6)
valloader = DataLoader(valdata,batch_size=batch_size,shuffle=True,  pin_memory=True, num_workers=6)


# In[8]:


#print(traindata.classes) # show all classes
#print(traindata.class_to_idx) # show the mapping from class to index.


# In[9]:


idx_to_class = {val: key for key, val in traindata.class_to_idx.items()} # build an inverse mapping for later use
#print(idx_to_class)


# In[10]:


correct_idx2class = {9: 'Frog1', 10: 'Frog2', 19: 'Frog3', 3: 'Grylloidea1', 14: 'Grylloidea2', 0: 'Tettigonioidea1', 1: 'Tettigonioidea2', 11: 'drums_FloorTom', 5: 'drums_HiHat', 6: 'drums_Kick', 4: 'drums_MidTom', 16: 'drums_Ride', 13: 'drums_Rim', 7: 'drums_SmallTom', 2: 'drums_Snare', 15: 'guitar_3rd_fret', 12: 'guitar_7th_fret', 18: 'guitar_9th_fret', 17: 'guitar_chord1', 8: 'guitar_chord2'}
#print(correct_idx2class)


# In[11]:


correct_class2idx = {val: key for key, val in correct_idx2class.items()}
#print(correct_class2idx)


# In[12]:


corrected_idx2idx = {val: correct_class2idx[key] for key, val in traindata.class_to_idx.items()}
#print(corrected_idx2idx)


# # Build an example network
# If you're unfamiliar with this part, please see the HW1 tutorial.

# In[13]:


import utils.resnet as resnet
model =resnet.resnet50(num_classes= len(traindata.classes))


# # Load model

# In[24]:


def load_model(model,filename):
    model.load_state_dict(torch.load(filename))
    return model
#net = Net(num_classes=len(traindata.classes)) # initialize your network
net = model
net = load_model(net, MODEL_PATH)
# Whether to use GPU or not?
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'
print("use",device,"now!")
net.to(device)


# # Evaluate on validation data

# In[43]:


net.eval()
correct = 0
vresult = []
vtarget = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(valloader):
        #print(type(data))
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        pred = output.data.max(1, keepdim=True)[1]
        
        vresult = vresult + list(pred.cpu().numpy().ravel())
        vtarget = vtarget + list(target.cpu().numpy().ravel())
        
        correct += pred.eq(target.data.view_as(pred)).sum()
    acc = correct.item() / len(valloader.dataset)
print("Validation Classification Accuracy: %f"%(acc))



