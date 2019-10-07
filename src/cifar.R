cifar <- dataset_cifar10()

cifar_x_train <- cifar$train$x
cifar_x_test <- cifar$test$x

#visualisation
i = 7777
val = rgb(cifar$train$x[i,,,1]/255,cifar$train$x[i,,,2]/255,cifar$train$x[i,,,3]/255)
dim(val)=c(32,32)
grid.raster(val)
grid.raster(val, interpolate=FALSE)

#reshape
dim(cifar_x_train) <- c(nrow(cifar_x_train),3072)
dim(cifar_x_test) <- c(nrow(cifar_x_test),3072)

#rescale
cifar_x_train <- cifar_x_train/255
cifar_x_test <- cifar_x_test/255

#Y
cifar_y_train <- cifar$train$y
cifar_y_test <- cifar$test$y
cifar_y_train <- to_categorical(cifar_y_train)
cifar_y_test <- to_categorical(cifar_y_test)

#model Dense 

cifar_model <- keras_model_sequential()
cifar_model <- layer_dense(cifar_model, units = 512, input_shape = 3072, activation ="relu")
cifar_model <- layer_dropout(cifar_model,0.3)
cifar_model <- layer_dense(cifar_model,units = 256, activation = "relu")
cifar_model <- layer_dropout(cifar_model, 0.1)
cifar_model <- layer_dense(cifar_model,units = 128, activation = "relu")
cifar_model <- layer_dense(cifar_model,units = 10, activation = "softmax")

compile(cifar_model, loss= "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
fit(cifar_model, cifar_x_train[1:12500,], cifar_y_train[1:12500,], epochs=50, batch_size = 20)

dim(cifar_x_train)
dim(cifar_y_train)

evaluate(cifar_model,cifar_x_train,cifar_y_train)

#modele de convolution

convo_model <- keras_model_sequential()
convol_model <- layer_conv_2d(convo_model,kernel_size = c(4,4),strides = 1,filters = 8,padding ="same",activation="relu",input_shape = c(32,32,3))
convo_model <- layer_locally_connected_2d(convo_model, filters=4, kernel_size = c(3,3)) #optionnel
convol_model <- layer_max_pooling_2d(convo_model, pool_size = c(2,2), stride = 2)
convo_model <- layer_dropout(convo_model, rate=0.3)
convo_model <- layer_flatten(convo_model)
convo_model <- layer_dense(convo_model, units = 50, activation = "relu")
convo_model <- layer_dropout(convo_model, rate=0.2)
convo_model <- layer_dense(convo_model, activation="softmax", units= 10)

cifar_convo <- dataset_cifar10()
cifar_convo_x_train <- cifar_convo$train$x/255
cifar_convo_x_test <- cifar_convo$test$x/255


cifar_convo_y_train <- cifar_convo$train$y
cifar_convo_y_test <- cifar_convo$test$y
cifar_convo_y_train <- to_categorical(cifar_convo_y_train)
cifar_convo_y_test <- to_categorical(cifar_convo_y_test)

compile(convo_model, loss= "categorical_crossentropy", optimizer = "rmsprop", metrics = "accuracy")
fit(convo_model, cifar_convo_x_train[1:5000,,,], cifar_convo_y_train[1:5000,], epochs = 50, batch_size = 10,validation_split = 0.3)

evaluate(object = convo_model, cifar_convo_x_train, cifar_convo_y_train)
evaluate(object=convo_model, cifar_convo_x_test, cifar_convo_y_test)
