#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import zipfile
import shutil
import pandas as pd
import requests
import numpy as np
import operator
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets,svm,metrics,model_selection


# In[9]:



os.mkdir('C:/Users/17677/Desktop/data/image')

def remove_file(old_path, new_path):
    filelist = os.listdir(old_path) #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        shutil.move(src, dst)

if __name__ == '__main__':
    remove_file(r"C:/Users/17677/Desktop/data/image1", r"C:/Users/17677/Desktop/data/image")
    remove_file(r"C:/Users/17677/Desktop/data/image2", r"C:/Users/17677/Desktop/data/image")
    remove_file(r"C:/Users/17677/Desktop/data/image3", r"C:/Users/17677/Desktop/data/image")


# In[20]:


#把所有照片放在一个文件夹中

determination = 'C:/Users/17677/Desktop/data/images'
if not os.path.exists(determination):
    os.makedirs(determination)

#源文件夹路径
path = 'C:/Users/17677/Desktop/data'
folders= os.listdir(path)
for folder in folders:
    dir = path + '/' +  str(folder)
    files = os.listdir(dir)
    for file in files:
        source = dir + '/' + str(file)
        deter = determination + '/'+ str(file)
        shutil.move(source, deter)


# In[21]:


label = pd.read_csv('C:/Users/17677/Desktop/label.csv',header=None,names = ['id_code','diagnosis'])
label = label.drop(label.index[[0]])
print(label.head(5))
l = len(label)
print(type(int(label.diagnosis[2])))
 os.makedirs('C:/Users/17677/Desktop/data/image/image0')
    os.makedirs('C:/Users/17677/Desktop/data/image/image1')
    os.makedirs('C:/Users/17677/Desktop/data/image/image2')
    os.makedirs('C:/Users/17677/Desktop/data/image/image3')
    os.makedirs('C:/Users/17677/Desktop/data/image/image4')
for i in range(l):
    i = i+1
    root='C:/Users/17677/Desktop/data/images'+'/'+str(label.id_code[i]) + ".png"
    if os.path.exists(root):
        if int(label.diagnosis[i])==0:
            shutil.move('C:/Users/17677/Desktop/data/images'+'/'+str(label.id_code[i]) + ".png","C:/Users/17677/Desktop/data/image/image0"+'/'+str(label.id_code[i]) + ".png") 
        if int(label.diagnosis[i])==1:
            shutil.move('C:/Users/17677/Desktop/data/images'+'/'+str(label.id_code[i]) + ".png","C:/Users/17677/Desktop/data/image/image1"+ '/'+str(label.id_code[i]) +".png") 
        if int(label.diagnosis[i])==2:
            shutil.move('C:/Users/17677/Desktop/data/images'+'/'+str(label.id_code[i]) + ".png","C:/Users/17677/Desktop/data/image/image2"+'/' +str(label.id_code[i]) +".png") 
        if int(label.diagnosis[i])==3:
            shutil.move('C:/Users/17677/Desktop/data/images'+'/'+str(label.id_code[i]) + ".png","C:/Users/17677/Desktop/data/image/image3"+'/'+ str(label.id_code[i]) +".png")
        if int(label.diagnosis[i])==4:
            shutil.move('C:/Users/17677/Desktop/data/images'+'/'+str(label.id_code[i]) + ".png","C:/Users/17677/Desktop/data/image/image4"+'/'+ str(label.id_code[i]) +".png") 


# In[25]:


import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets,svm,metrics,model_selection
from sklearn.preprocessing import StandardScaler


# In[26]:


path = 'C:/Users/17677/Desktop/data/image'
categorys = os.listdir(path)

X = []
Y_label= []
for category in categorys:
    images = os.listdir(path+'/'+category)
    for image in images:
        im = ft.hog(Image.open(path+'/'+category+'/'+image).convert('L').crop((64,128,320,384)),
                    orientations=9, 
                    pixels_per_cell=(16,16), 
                    cells_per_block=(2,2), 
                    block_norm = 'L2-Hys', 
                    transform_sqrt = True, 
                    feature_vector=True, 
                    visualize=False
                    )
        X.append(im)
        Y_label.append(category)        
X = np.array(X)
print(X.shape)
Y_label = np.array(Y_label)
Y = LabelEncoder().fit_transform(Y_label)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=1)
#归一化
standardscaler=StandardScaler()
standardscaler.fit(X_train)
X_train=standardscaler.transform(X_train)
X_test=standardscaler.transform(X_test)


# In[27]:


import matplotlib as mpl
import matplotlib.pyplot as plt
standardscaler=StandardScaler()
standardscaler.fit(X_train)
X_train=standardscaler.transform(X_train)
X_test=standardscaler.transform(X_test)
plt.scatter(X_train[y_train==0][:,0],X_train[y_train==0][:,1],color='r')
plt.scatter(X_train[y_train==1][:,0],X_train[y_train==1][:,1],color='g')
plt.scatter(X_train[y_train==2][:,0],X_train[y_train==2][:,1],color='b')
plt.scatter(X_train[y_train==3][:,0],X_train[y_train==3][:,1],color='y')
plt.scatter(X_train[y_train==4][:,0],X_train[y_train==4][:,1],color='m')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.savefig('C:/Users/17677/Desktop/test7.jpg')
plt.show()


# In[28]:


from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd

##借鉴自CSDN
class Knn():
    def __init__(self,k=5):
        self.k = k

    def fit(self,x,y):
        self.x = x
        self.y = y

    def dis(self,instant1, instant2):  # 求欧式距离
        dist = np.sqrt(np.sum((instant1 - instant2) ** 2))
        return dist

    def knn_classfy(self,X,y,test):
        distances = [self.dis(x,test) for x in X]
        kneighbors = np.argsort(distances)[:self.k] #选取前K个邻居
        count = Counter(y[kneighbors]) #记标签的个数
        return count.most_common()[0][0]  #统计出现次数最多的标签
    
    def predict(self,test_x):
        pre = [self.knn_classfy(self.x,self.y,i) for i in test_x]  # 预测的标签集
        return  pre

    def score(self,test_x,test_y):
        pre = [self.knn_classfy(self.x,self.y,i) for i in test_x]  # 预测的标签集
        col = np.count_nonzero((pre == test_y))  # 统计测试集的标签和预测的标签相同的个数
        return col / test_y.size
##
accuracy = [];
for j in range(20):
    mknn = Knn(j+1)
    print("近邻值：",(j+1))
    mknn.fit(X_train, y_train)
    predictions_labels = list(mknn.predict(X_test))
    labels = ["0","1","2","3","4"]

    print("准确率：",mknn.score(X_test, y_test))
    accuracy.append(mknn.score(X_test, y_test));
    print(confusion_matrix(y_test, predictions_labels))
    print (classification_report(y_test, predictions_labels))
    print(accuracy)
    


# In[29]:


import matplotlib.pyplot as plt
 
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
y = accuracy;
plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(x, y, marker='o', mec='r', mfc='w')
#plt.plot(x, y, marker='*', ms=10)
plt.savefig('C:/Users/17677/Desktop/test4.jpg')
plt.show()


# In[ ]:





# In[ ]:




