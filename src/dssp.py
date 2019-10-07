import keras
import numpy as np
from keras import Sequential
from sklearn.preprocessing import LabelEncoder #pour le one_hot_encoding.
from keras.layers import Dense, Flatten, TimeDistributed
from keras import Input, Model
from keras.layers import add, Activation #add pour sommer les couches
from keras.layers import Conv1D, AveragePooling1D
from keras.models import load_model
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint

#Training set
x_train = []

with open("./train.fasta", "r") as file:
    for line in file:
        x_train.append(list(line[:-1]))

for seq in x_train:
    seq.append("J")
    while len(seq) != 760:
        seq.append("J")

y_train = []

with open("./train.dssp", 'r') as file:
    for line in file:
        y_train.append(list(line[:-1]))

for d in y_train:
    d.append("*")
    while len(d) != 760:
        d.append("*")

all_one_hot_x_train = []

for i in range(0,len(x_train)):
    classes = LabelEncoder()
    integer_encoded = classes.fit_transform(x_train[i])
    one_hot_i = keras.utils.to_categorical(integer_encoded, num_classes= 21)  
    all_one_hot_x_train.append(one_hot_i)
all_one_hot_x_train = np.array(all_one_hot_x_train)

all_one_hot_y_train = []

for j in range(0,len(y_train)):
    classes = LabelEncoder()
    integer_encoded = classes.fit_transform(y_train[j])
    one_hot_j = keras.utils.to_categorical(integer_encoded, num_classes= 4)
    all_one_hot_y_train.append(one_hot_j)
all_one_hot_y_train = np.array(all_one_hot_y_train)

#Testing set
x_test = []

with open("./blind.fasta", "r") as file:
    for line in file:
        x_test.append(list(line[:-1]))

for seq in x_test:
    seq.append("J")
    while len(seq) != 760:
        seq.append("J")

y_test = []

with open("./blind.dssp", 'r') as file:
    for line in file:
        y_test.append(list(line[:-1]))

for d in y_test:
    d.append("*")
    while len(d) != 760:
        d.append("*")

all_one_hot_x_test = []

for i in range(0,len(x_test)):
    classes = LabelEncoder()
    integer_encoded = classes.fit_transform(x_test[i])
    one_hot_i = keras.utils.to_categorical(integer_encoded, num_classes = 21)  
    all_one_hot_x_test.append(one_hot_i)
all_one_hot_x_test = np.array(all_one_hot_x_test)

all_one_hot_y_test = []

for j in range(0,len(y_test)):
    classes = LabelEncoder()
    integer_encoded = classes.fit_transform(y_test[j])
    one_hot_j = keras.utils.to_categorical(integer_encoded, num_classes = 4)  
    all_one_hot_y_test.append(one_hot_j)
all_one_hot_y_test = np.array(all_one_hot_y_test)

#Creation du modele CNN
model = Sequential()
model.add(Conv1D(filters=21, kernel_size=3, strides=1, padding= "same", activation = "relu", kernel_initializer="he_normal"))
model.add(Conv1D(filters=21, kernel_size=3, strides=1, padding= "same", activation = "linear", kernel_initializer="he_normal"))
model.add(TimeDistributed(Dense(4,activation = "softmax")))

np.random.seed(1000)
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
stop_criteria =  EarlyStopping(monitor="val_loss", mode = "min",verbose = 1, patience = 10)
best_model_path = "./my_model"+".h5"
best_model = ModelCheckpoint(best_model_path, monitor = "val_loss", verbose = 2, save_best_only = True)
my_model = load_model("./my_model.h5")
my_model.fit(all_one_hot_x_train,all_one_hot_y_train,epochs=20, batch_size = 20,
          validation_split = 0.05, callbacks = [stop_criteria])

evaluation = model.evaluate(all_one_hot_x_train, all_one_hot_y_train)
print(evaluation)

predictions = my_model.predict(all_one_hot_x_test)

tp = 0
tn = 0
fn = 0
fp = 0
tot = 0
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        if all_one_hot_y_test[i,j,3] != 0:
            predmax = -1
            predict_class = -1
            true_class = -1
            for k in range(len(predictions[i,j])):
                if predmax < predictions[i,j,k]:
                    predmax = predictions[i,j,k]
                    predict_class=k
                if all_one_hot_y_test[i,j,k] == 1.:
                    true_class = k
            if predict_class == true_class:
                tp = tp+1
            tot=tot+1
print("tp =", tp)
print("tot =", tot)
print("ACC:%5.3f"%(tp/tot*100))

#creation du modele Resnet
def residual_module(lay_i, n_filters):
    if lay_i.shape[-1] != n_filters:
        lay_i = Conv1D(n_filters, (1), padding="same", activation="relu",kernel_initializer="he_normal")(lay_i)
    save = lay_i
    conv_1 = Conv1D(n_filters, (3), padding="same", activation="relu",kernel_initializer="he_normal")(lay_i)
    conv_2 = Conv1D(n_filters, (3), padding="same", activation="linear",kernel_initializer="he_normal")(conv_1)
    conc_1 = add([conv_2, save])
    output = Activation("relu")(conc_1)

    return output

def my_model():
    n_residual = 5
    print("Simple residual network with {} modules".format(n_residual))
    inputs = Input(shape=(760, 21))
    residual_i = inputs
    for _ in range(n_residual):
        residual_i = residual_module(residual_i, 24)

    output = TimeDistributed(Dense(4, activation="softmax"))(residual_i)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

resnet = my_model()
resnet.fit(all_one_hot_x_train, all_one_hot_y_train, batch_size=20, epochs=20)
resnet.evaluate(all_one_hot_x_test, all_one_hot_y_test)

predictions = resnet.predict(all_one_hot_x_test)

tp = 0
tn = 0
fn = 0
fp = 0
tot = 0
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        if all_one_hot_y_test[i,j,3] != 0:
            predmax = -1
            predict_class = -1
            true_class = -1
            for k in range(len(predictions[i,j])):
                if predmax < predictions[i,j,k]:
                    predmax = predictions[i,j,k]
                    predict_class=k
                if all_one_hot_y_test[i,j,k] == 1.:
                    true_class = k
            if predict_class == true_class:
                tp = tp+1
            tot=tot+1
print("tp =", tp)
print("tot =", tot)
print("ACC:%5.3f"%(tp/tot*100))
