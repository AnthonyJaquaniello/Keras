from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
         'Uniformity of Cell Shape','Marginal Adhesion',
         'Single Epithelial Cell Size','Bare Nuclei',
         'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'class']
dataset = pd.read_csv("../data/breast-cancer-wisconsin.data",names=names)

output_check = dataset.applymap(np.isreal).all(0)
for i in output_check:
    if (i == False):
        print("Error")
        print(output_check)
        quit()

dataset["class"] = dataset["class"].astype("category")
dataset["class-value"] = dataset["class"].cat.codes
#conversion en float
dataset["class-value"].astype(float)

X = dataset[['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']]
X = X.to_numpy()  #transform en numpy pour keras

Y = dataset[['class-value']]  #Recupere colonne correspondante
Y = Y.values #Recupere les valeurs
Y = Y.astype(float)

classes = LabelEncoder()
classes.fit(Y)
classes_Y = classes.transform(Y)

model = Sequential()
model.add(Dense(10,input_dim=9 , activation ="sigmoid"))
model.add(Dense(4,activation ="sigmoid")) #couche cachée 
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
model.fit(X,Y,epochs=25, batch_size = 10)

evaluation = model.evaluate(X,Y) 
print("accuracy",(evaluation[1]*100))
print("loss",(evaluation[0]))

predictions = model.predict(X)

for i in range(10):
    distance = Y[i]-predictions[i]
    print("%s => %d (expected %d)[D=%f]"% (X[i].tolist(), predictions[i], Y[i], distance  ))


