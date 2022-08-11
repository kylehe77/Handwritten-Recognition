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
from dict_related import dict_related
import numpy as np
import cv2
import random,os
import time


"""
Reference code：
https://blog.csdn.net/qq_51938362/article/details/122073259
https://blog.csdn.net/qq_40211493/article/details/106580655
https://blog.csdn.net/Chris_zhangrx/article/details/86516331
"""





train_datasets_len =  697984 ##when the default batch size is 64, len(traning datasets of EMNIST)/batch_size; and because the result of it is not a whole number, plus 1. then *batch size
                             ## we want to make the progress bar would not exceed 100%


#set train_datasets_len as a global variable, when the batch size is changed , then train_datasets_len will be recalculated.
def get_train_datasets_len(batch_size=64):
    global train_datasets_len
    really_train_datasets_len = 697932
    train_datasets_len = (really_train_datasets_len//batch_size + 1)*batch_size
    return train_datasets_len

class DNN(nn.Module):
  
    def __init__(self):
        super(DNN, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 96)
        self.l5 = torch.nn.Linear(96, 62)
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)



def down_datasets():
   
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  
        torchvision.transforms.Normalize((0.1307,),(0.3081,)) #increase accuracy, decrease loss
    ])
    
    try:
        DOWNLOAD_MNIST = False  #If EMNIST Dataset has been downloaded before, download should be set to false.And it can be found in ./data
        train_data = torchvision.datasets.EMNIST(root='./data',train=True,transform=transforms,download = DOWNLOAD_MNIST, split = 'byclass' )
        test_data = torchvision.datasets.EMNIST(root='./data',train=False,transform=transforms,download = DOWNLOAD_MNIST, split = 'byclass' )
        print ("Datasets already downloaded!")
    except:
        DOWNLOAD_MNIST = True  #If EMNIST Dataset has been downloaded before, download should be set to false.And it can be found in ./data
        train_data = torchvision.datasets.EMNIST(root='./data',train=True,transform= torchvision.transforms.ToTensor(),download = DOWNLOAD_MNIST, split = 'byclass' )
        test_data = torchvision.datasets.EMNIST(root='./data',train=False,transform= torchvision.transforms.ToTensor(),download = DOWNLOAD_MNIST, split = 'byclass' )
        print ("Datasets download finish!")
    return train_data,test_data

def DataLoader_set(train_data,test_data,batch_size):
    
    #Loading dataset
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader( dataset=test_data,  batch_size=batch_size , shuffle=True)
    return train_loader,test_loader




def train_dnn_model(train_datasets,test_datasets,dict_train):
    print("Start Training")
    # BATCH_SIZE = 64
    # EPOCHES = 5
    # LEARN_RATE = 6e-4
    #set a defult number for these elements
    BATCH_SIZE = dict_train["BATCH_SIZE"]
    EPOCHS = dict_train["EPOCHS"]
    LEARN_RATE = dict_train["LEARN_RATE"]
    MODEL_NAME = dict_train["model_name"]
    # Create a directory
    if (not os.path.exists("models")):
        os.mkdir("models")

    train_datasets_len = get_train_datasets_len(BATCH_SIZE)
    model = DNN()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= LEARN_RATE, momentum= 0.5)
    
    for epoch in range(EPOCHS):
        runing_loss = 0.0
        running_correct = 0.0
        t1 = time.time()
        for batch_idx, data in enumerate(train_datasets, 0):
            inputs, target = data

            optimizer.zero_grad()
            
            outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)

            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()

            
            runing_loss += loss.item()
            running_correct += torch.sum(pred == target)
            
            print("Train Epoch:%d | Batch Status: %d/%d (%.2f%%) | Loss: %.5f" % (epoch +1, (batch_idx+1)*BATCH_SIZE,train_datasets_len,(100*(batch_idx+1)*BATCH_SIZE/train_datasets_len),loss.item()))
            

        t2 = time.time()
        print ("Train time:{}\n".format(t2-t1))

        total =0
        correct = 0
        t1 = time.time()
        with torch.no_grad():
            print ("Runing Test ...")
            for data in test_datasets:
                images, labels =data
                outputs = model(images)  
                #Returns two values, the first is the maximum value and the second is the index of the maximum value. dim = 1 means the above result in the column dimension, dim = 0 means the above result in the row dimension.   
                _, predicted = torch.max(outputs.data, dim =1 ) 
                #Each labels in batch_size is a tuple of (N, 1), size(0) = N
                total += labels.size(0)  
                #the total number of correctly
                correct +=(predicted == labels).sum().item()  
        t2 = time.time()
        print("Test set:Average loss:%.4f Average accuracy %d %%" % (runing_loss/(batch_idx+1),100*correct/total))
        print ("Test time:{}\n".format(t2-t1))
        if correct/total>=0.85:##save the model if the accuracy is higher than 85%
            torch.save(model.state_dict(), MODEL_NAME)
        else:
            torch.save(model.state_dict(), MODEL_NAME.split(" ")[0]+"_"+str(epoch+1)+"."+MODEL_NAME[-1])
    torch.save(model.state_dict(), MODEL_NAME)
    print("End Train Model")



def predict_image_singer_v1(dict_predict={}):
    # 
    # The original image was rotated by 90 degrees, so the standard image was predicted incorrectly, while the rotated  image was predicted correctly; 
    # Non-standard images do not need to be converted; the downloaded image is a non-standardized image, which has been rotated
    # 
    print (dict_predict["model_name"])
    try:
        image = Image.open(dict_predict["file_path"]) 
        image = np.array(image).transpose(1,0)
        data_dataset = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  
            torchvision.transforms.Normalize((0.1307,),(0.3081,)) #increase accuracy, decrease loss
        ])
        my_tensor = data_dataset(image)
       
        my_tensor = my_tensor.resize_(28,28)
        net = DNN()
        net.load_state_dict(torch.load(dict_predict["model_name"]))##load the specific model to predict digts/letters
        net.eval()#set dropout and batch normalization layers to evaluation mode before running inference

        pred = net(my_tensor)#input the image to get ouput from net
        probability = float(nn.functional.softmax(pred.data,dim=1).max(1).values)
        predicted = int(pred.argmax(1))
        
        return probability,dict_related[str(predicted)]
    except Exception as e:
        print ('The image must be single channel')
        return "Format Unupport","-1"



