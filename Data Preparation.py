#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle


img_dim = 32
n = 43

######################### Loading Training Data ########################
train_path = 'Data/Training/'

n_path = [] 
for i in range(n):
    if len(str(i)) == 1:
        n_path.append(f'0000{i}')
    else:
        n_path.append(f'000{i}')

X_train = []
y_train = []

for class_folder in n_path:
    ## reading the annotations file
    img_name = []
    for i in os.listdir(f'{train_path}{class_folder}'):
        if i.endswith('.csv'):
            annot_path=i
        elif i.endswith('.ppm'):
            img_name.append(i)

    annot_df = pd.read_csv(f'{train_path}{class_folder}/{annot_path}', sep=';')


    for img_path in img_name:
        x1, y1, x2, y2, label = annot_df[annot_df['Filename']==img_path].iloc[:,3:].values[0]
        img = Image.open(f'{train_path}{class_folder}/{img_path}')
        img = img.crop((x1, y1, x2, y2))
        img = img.resize((img_dim, img_dim))
        img = np.array(img)

        X_train.append(img)
        y_train.append(label)
        
with open('Data/train.pkl','wb') as f:
    pickle.dump((X_train,y_train), f)


######################### Loading Testing Data ########################

test_path = 'Data/Testing/'

df = pd.read_csv(f'{test_path}Test.csv')
img_dim = 32

X_test = []
y_test = []

for j in os.listdir(test_path):       
    if j.endswith('.png'):
        x1,y1,x2,y2,label = df[df['Path'] == f'Test/{j}'].iloc[:,2:-1].values[0]

        img = Image.open(f'{test_path}{j}')
        img = img.crop((x1, y1, x2, y2))
        img = img.resize((img_dim, img_dim))
        img = np.array(img)

        X_test.append(img)
        y_test.append(label)


with open('Data/test.pkl','wb') as f:
    pickle.dump((X_test,y_test), f)


# In[ ]:




