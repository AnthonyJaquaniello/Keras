import numpy as np
import pandas 
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils import np_utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LabelEncoder

file_data= "breast-cancer-wisconsin.data"
names=['Sample code' ,'Clump thickness','Uniformity of Cell size',
'Uniformity of cell shapes','Marginal Adhesion','Single Epithelial Cell size',
'Bare nuclei','Bland chromatin','Normal nucleoli','Mitoses','class']
rawdataset = pandas.read_csv(file_data, names=names)


output_check = rawdataset.applymap(np.isreal).all(0)
for i in output_check:
	if (i == False):
		print("Error")
		print(output_check)
		quit()



rawdataset["class"]=rawdataset["class"].astype("category")
rawdataset["class-value"]=rawdataset["class"].cat.codes
rawdataset["class-value"].astype(float)

print(rawdataset.shape)
print(rawdataset.head(20))
print(rawdataset.describe())
print(rawdataset.groupby("class").size())

X = rawdataset[['Clump thickness']]
X = X.to_numpy()  #transform en numpy pour keras

Y = rawdataset[['class-value']]  #Recupere colonne correspondante
Y = Y.values #Recupere les valeurs
Y = Y.astype(float) #

print("Data values")
print(X)
print(Y)

classes = LabelEncoder()
classes.fit(Y)
classes_Y = classes.transform(Y)

onehot_Y = np_utils.to_categorical(classes_Y)

print(X)
print(Y)

print(X.shape)
print(Y.shape)

model = Sequential()
model.add(Dense(10,input_dim =8 , activation ="sigmoid")) #taille = 10
model.add(Dense(4,activation ="sigmoid"))#couche cachée 
model.add(Dense(1, activation="sigmoid"))	#et 1 couche de sortie

#compiler le modele Los = calcule l'erreur
model.compile("binary_crossentropy",optimizer="adam", metrics=['accuracy'])
model.fit(X,Y,epochs=1, batch_size = 10) #epochs = nombre iterations, va tout afficher
#Evaluer le modele
evaluation = model.evaluate(X,Y) 
print("accuracy"%(evaluation[1]*100))
print("loss"%(evaluation[0]))

#Chiffre brut de la prediction
predictions = model.predict(X)
print("raw prediction")
for i in range(10):
	print(predictions[i])

rounded = 


print( keras.__version)
print( keras.__version)
