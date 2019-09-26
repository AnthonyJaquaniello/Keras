#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:54:44 2019

@author: anthony_jaquaniello

Residual network implementation

Input: suite de matrices (20*400) (one hot encoding)
       avec (aa,position_sequence)
"""

import keras
import numpy as np

from sklearn.preprocessing import LabelEncoder #pour le one_hot_encoding.
from keras.layers import Dense, Flatten #Flatten pour mettre à plat une matrice.
from keras import Input, Model
from keras.layers import add, Activation #add pour sommer les couches
#from keras.utils import plot_model  # Needs pydot.
from keras.layers import Conv1D, AveragePooling1D #pour le pooling et la convolution.


def residual_module(lay_i, n_filters):
    """
    Fonction de création de module.
    """
    save = lay_i #on stocke la première couche dans une variable.
    # check if the number of filters needs to be increased, assumes
    # channels last format.
#    if lay_i.shape[-1] != n_filters:
#        lay_i = Conv1D(n_filters, (1), padding="same", activation="relu",
#                       kernel_initializer="he_normal")
    conv_1 = Conv1D(n_filters, (3), padding="same", activation="relu",
                    kernel_initializer="he_normal")(lay_i)
    conv_2 = Conv1D(n_filters, (3), padding="same", activation="linear",
                    kernel_initializer="he_normal")(conv_1)
    conc_1 = add([conv_2, save])
    output = Activation("relu")(conc_1)

    return output

def my_model7():
    n_residual = 4 #on chaine 4 fois les modules
    print("Simple residual network with {} modules".format(n_residual))
    inputs = Input(shape=(400, 20))
    residual_i = inputs
    for _ in range(n_residual):
        residual_i = residual_module(residual_i, 20)

    gavg_1 = AveragePooling1D((2), strides=(1))(residual_i)
    flat_1 = Flatten()(gavg_1)
    output = Dense(10, activation="softmax")(flat_1)

    model = Model(inputs=inputs, outputs=output)
    #Pour l'API fonctionnelle

#    plot_model(model, to_file="residual.png",
#               show_shapes=True, show_layer_names=True)
    print(model.summary())

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# Data treatment: traininig data.
file_data_train = "../data/reduced_train.npz"

train = np.load(file_data_train)
X_train = train["X_train"]
Y = train["y_train"]
mask_train = train["mask_train"]

# Print these to visualize.
print(X_train.shape)
print(Y.shape)
print(mask_train.shape)

# One hot encode the output.
classes = LabelEncoder()
classes.fit(Y)
classes_Y = classes.transform(Y)

onehot_Y = keras.utils.to_categorical(classes_Y)

# Creating the model.
model = my_model7()
# Training the model;
model.fit(X_train, onehot_Y, batch_size=20, epochs=20)

# Perf of the model.
model.evaluate(X_train, onehot_Y)


# Validation data.
file_data_val = "../data/reduced_val.npz"

val = np.load(file_data_val)
X_val = val["X_val"]
y_val = val["y_val"]
mask_val = val["mask_val"]

# One hot encode.
y_val = keras.utils.to_categorical(y_val)

model.evaluate(X_val, y_val)
