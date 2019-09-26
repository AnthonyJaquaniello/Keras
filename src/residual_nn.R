library(keras)
library(reticulate)
np <- import("numpy")

reduced_train <- np$load('../data/reduced_train.npz')

x_train <- reduced_train$f[["X_train.npy"]]
y_train <- reduced_train$f[["y_train.npy"]]


#on a du conv1d cette fois (pour les données reduced_train)
#D'abord on va commencer doux, par un modèle simple
#Avec layer_add, le résultat d'une couche est additionné a resultat de la couche d'avant

dim(x_train)
dim(y_train)

dim(x_train) <- c(nrow(x_train), 400, 20)
y_train <- to_categorical(y_train)

residual_model <- function(lay, n_filters){
  save <- lay
  conv_1 <- layer_conv_1d(lay, filters = n_filters, padding = "same", activation = "relu", kernel_size = c(3,3), input_shape = c(400,20), kernel_initializer = "he_normal")
  conv_2 <- layer_conv(conv_1, filters = n_filters, padding = "same", activation = "linear", kernel_size = c(3,3), kernel_initializer = "he_normal")
  conv_model <- add(list[conv_2,lay])
  output = activation_relu(conv_model)
}

the_model <- function(){
  n_residual = 2
  for(i in 1:n_residual){
    residual_i
  }
}

