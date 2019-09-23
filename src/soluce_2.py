#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas 
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils import np_utils

#scikit-learn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


# In[26]:


file_data= "breast-cancer-wisconsin.data"
names=['Sample code' ,'Clump thickness','Uniformity of Cell size',
'Uniformity of cell shapes','Marginal Adhesion','Single Epithelial Cell size',
'Bare nuclei','Bland chromatin','Normal nucleoli','Mitoses','class']
rawdataset = pandas.read_csv(file_data, names=names)


# In[27]:


output_check = rawdataset.applymap(np.isreal).all(0)
for i in output_check:
	if (i == False):
		print("Error")
		print(output_check)
		quit()


# In[28]:



rawdataset["class"]=rawdataset["class"].astype("category")
rawdataset["class-value"]=rawdataset["class"].cat.codes
rawdataset["class-value"].astype(float)


# In[29]:


print("Shape of dataset")
print(rawdataset.shape)
print("First 20 lines of dataset")
print(rawdataset.head(20))
print("Summary of dataset")
print(rawdataset.describe())
print("Class distribution")
print(rawdataset.groupby("class").size())


# In[32]:


X = rawdataset[['Clump thickness','Uniformity of Cell size',
'Uniformity of cell shapes','Marginal Adhesion','Single Epithelial Cell size',
'Bare nuclei','Bland chromatin','Normal nucleoli','Mitoses']]

X = X.to_numpy()  #transform en numpy pour keras


# In[33]:


Y = rawdataset[['class-value']]  #Recupere colonne correspondante
Y = Y.values #Recupere les valeurs
Y = Y.astype(float) #


# In[34]:


print("Data values")
print(X)
print(Y)


# In[35]:


classes = LabelEncoder()
classes.fit(Y)
classes_Y = classes.transform(Y)

#onehot_Y = np_utils.to_categorical(classes_Y)


# In[36]:


print(X)
print(Y)


# In[37]:


print(X.shape)
print(Y.shape)


# In[38]:


model = Sequential()


# In[39]:


model.add(Dense(10,input_dim =9 , activation ="sigmoid")) #taille = 10


# In[40]:


model.add(Dense(4,activation ="sigmoid"))#couche cachée 


# In[41]:


model.add(Dense(1, activation="sigmoid"))	#et 1 couche de sortie


# In[43]:


#compiler le modele Los = calcule l'erreur
model.compile(loss = "binary_crossentropy",optimizer="adam", metrics=['accuracy'])


# In[49]:


model.fit(X,Y,epochs=30, batch_size = 10) #epochs = nombre iterations, va tout afficher
print(model.summary())


# In[52]:


#Evaluer le modele
evaluation = model.evaluate(X,Y) 
print("accuracy %.2f"%(evaluation[1]*100))
print("loss  %.2f "%(evaluation[0]))


# In[51]:


#Chiffre brut de la prediction
predictions = model.predict(X)
print("raw prediction")
for i in range(10):
	print(predictions[i])


# In[55]:


rounded = [round(x[0]) for x in predictions]
print('Rounded row predictions')
for i in range(10):
    print(rounded[i])
    


# In[58]:


#Summarize the first 10 cases
for i in range(10):
    distance = Y[i]-predictions[i]
    print("%s => %d (expected %d)[D=%f]"% (X[i].tolist(), predictions[i], Y[i], distance  ))

