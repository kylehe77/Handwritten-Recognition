import torch
import torchvision  
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import random,os
import uuid
import time
import json

'''
Reference code:
https://www.geeksforgeeks.org/json-load-in-python/
'''
emnist_byclass = 814255 ##the data of EMNIST dataset
train_byclass  = 697932 
test_byclass   = 116323

def down_datasets_v1():
    
    
    try:
        DOWNLOAD_MNIST = False  #If EMNIST Dataset has been downloaded before, download should be set to false.And it can be found in ./data
        train_data = torchvision.datasets.EMNIST(root='./data',train=True,download = DOWNLOAD_MNIST, split = 'byclass' )
        test_data = torchvision.datasets.EMNIST(root='./data',train=False,download = DOWNLOAD_MNIST, split = 'byclass' )
        print ("datasets download finish or exists!")
    except:
        DOWNLOAD_MNIST = True  #If EMNIST Dataset has been downloaded before, download should be set to false.And it can be found in ./data
        train_data = torchvision.datasets.EMNIST(root='./data',train=True,download = DOWNLOAD_MNIST, split = 'byclass' )
        test_data = torchvision.datasets.EMNIST(root='./data',train=False,download = DOWNLOAD_MNIST, split = 'byclass' )
        print ("datasets download finish!")
        
    return train_data,test_data


#get the corresponding number of each digit or letter
def get_number(letter):
    with open("cache/dict_my_class.txt","r") as f:
        all_data = json.load(f)

    return str(all_data[letter])

#this function is designed for the filter， which can find the specific digit or letter
def filter_datasets(filter_data,mode="train"):
    if mode == "train":
        with open("cache/train_datasets_position.txt","r") as f:
            all_data = json.load(f)
        index = random.choice(all_data[filter_data])
    else:
        with open("cache/test_datasets_position.txt","r") as f:
            all_data = json.load(f)
        index = random.choice(all_data[filter_data])
    return index

#this function is designed to see the content of images
def look_image_content(train_datasets,index):
  
    
    #print (index)
    for i,(img,label) in enumerate(train_datasets):
        if (index == i):
            image = img
            break
    #print(np.array(image).shape)
    image = np.array(image).squeeze().transpose(1,0)*255
    #print (image.shape,image)
    #display_image(image)
    cv2.imwrite("cache/temp.png",image) #create a cache file would improve the performance of the code
    
