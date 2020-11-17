#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import zipfile
import shutil
import pandas as pd
import requests
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder


# In[2]:


url = 'https://ml-cnn.obs.cn-north-4.myhuaweicloud.com:443/data/label.csv?AccessKeyId=0IDITRXGOTDIMCNK7RIZ&Expires=1593409348&Signature=Q2MUN7OYIIlvP1yCEZDcfZOjnmQ%3D'
r = requests.get(url)
with open('./label.csv', "wb") as code:
    code.write(r.content)
url1 = 'https://ml-cnn.obs.cn-north-4.myhuaweicloud.com:443/data/image2.zip?AccessKeyId=0IDITRXGOTDIMCNK7RIZ&Expires=1593408808&Signature=3s62oNOQwgdptA6BRjvcNiEV540%3D'
r = requests.get(url1)
with open('./image1.zip', "wb") as code:
    code.write(r.content)
url2 ='https://ml-cnn.obs.cn-north-4.myhuaweicloud.com:443/data/image1.zip?AccessKeyId=0IDITRXGOTDIMCNK7RIZ&Expires=1593408832&Signature=UQmE%2BYESlmQNuXFzZSSHVMHTW7c%3D'
r = requests.get(url2)
with open('./image2.zip', "wb") as code:
    code.write(r.content)
url3 = 'https://ml-cnn.obs.cn-north-4.myhuaweicloud.com:443/data/image3.zip?AccessKeyId=0IDITRXGOTDIMCNK7RIZ&Expires=1593408852&Signature=%2BSlgSAB1BKrRmMeCLmM8f/fPFuo%3D'
r = requests.get(url3)
with open('./image3.zip', "wb") as code:
    code.write(r.content)


# In[ ]:


dataset1 = zipfile.ZipFile('./image1.zip')
dataset1.extractall('./image')
dataset2 = zipfile.ZipFile('./image2.zip')
dataset2.extractall('./image')
dataset3 = zipfile.ZipFile('./image3.zip')
dataset3.extractall('./image')


# In[ ]:


#把所有照片放在一个文件夹中
determination = './images'
if not os.path.exists(determination):
    os.makedirs(determination)

#源文件夹路径
path = './image'
folders= os.listdir(path)
for folder in folders:
    dir = path + '/' +  str(folder)
    files = os.listdir(dir)
    for file in files:
        source = dir + '/' + str(file)
        deter = determination + '/'+ str(file)
        shutil.move(source, deter)


# In[ ]:


label = pd.read_csv('./label.csv',header=None,names = ['id_code','diagnosis'])
label = label.drop(label.index[[0]])
l = len(label)
for i in range(l):
    i = i+1
    if int(label.diagnosis[i])==0:
        shutil.move('./images'+'/'+str(label.id_code[i]) + ".png","./data/0"+ '/'+str(label.id_code[i]) +".png") 
    if int(label.diagnosis[i])==1:
        shutil.move('./images'+'/'+str(label.id_code[i]) + ".png","./data/1"+ '/'+str(label.id_code[i]) +".png") 
    if int(label.diagnosis[i])==2:
        shutil.move('./images'+'/'+str(label.id_code[i]) + ".png","./data/2"+'/' +str(label.id_code[i]) +".png") 
    if int(label.diagnosis[i])==3:
        shutil.move('./images'+'/'+str(label.id_code[i]) + ".png","./data/3"+'/'+ str(label.id_code[i]) +".png") 
    if int(label.diagnosis[i])==4:
        shutil.move('./images'+'/'+str(label.id_code[i]) + ".png","./data/4"+'/'+ str(label.id_code[i]) +".png") 


# In[7]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ColorJitter(),
                    transforms.ToTensor(),
                    normalize
                    ])
dataset = ImageFolder(root ='./data',transform = transform)


# In[8]:


#使用Resnet
#划分训练集和测试集
train_size = int(len(dataset)*0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataset = DataLoader(dataset = train_dataset,batch_size =256,shuffle = True)
test_dataset  = DataLoader(dataset = test_dataset,batch_size =256*3,shuffle = True)

#加载预训练的模型
model = models.resnet50(pretrained=False,num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)


# In[9]:


losses = []
acc = []


# In[10]:


n_epochs = 50
for epoch in range(n_epochs):    
    total=0    
    correct = 0  
    print("epoch {}/{}".format(epoch,n_epochs))    
    print("-"*10)    
    for data in train_dataset:        
        img,labels=data    
        labels = labels-1
        #img=img.view(img.size(0),-1)        
        img = Variable(img)     
        if torch.cuda.is_available():            
            img=img.cuda()            
            labels=labels.cuda() 
            model = model.cuda()
        else:            
            img=Variable(img)            
            labels=Variable(labels)        
        out=model(img)#得到前向传播的结果   
        loss=criterion(out,labels)#得到损失函数   
        losses.append(loss)
        print("loss=",loss)
        predicted =  out.argmax(dim=1)
        print(int((predicted == labels).sum())/(labels.size(0)))
        print_loss=loss.data.item()        
        optimizer.zero_grad()#归0梯度        
        loss.backward()#反向传播        
        optimizer.step()#优化      
        #scheduler.step()
        correct += (predicted == labels).sum().float()
        total += len(labels)
    accuracy = correct/total
    acc.append(accuracy)
    print("train_accuracy  = ",float(accuracy))


# In[ ]:


#测试过程
for i,(image,labels) in enumerate(test_dataset):
    model.eval()
    output = model(image)
    labels = labels-1
    loss = criterion(output,labels)
    prediction = output.argmax(dim=1)
    correct = (prediction == labels).sum().float()
    total = len(labels)
    accuracy = correct/total
    print("test_accuracy  = ",float(accuracy))

    


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(labels, output))
print(classification_report(labels, output))

