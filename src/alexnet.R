library(keras)

cifar <- dataset_cifar10()

x_train <- cifar$train$x
x_test <- cifar$test$x/255
y_train <- to_categorical(cifar$train$y)
y_test <- to_categorical(cifar$test$y)



conv_model <- keras_model_sequential()

#1ere conv
conv_model <- layer_conv_2d(conv_model, filters = 96, kernel_size = 6, strides = 4, padding = 'same', activation = "relu", input_shape=c(32,32,3))
conv_model <- layer_max_pooling_2d(conv_model, pool_size = 2, padding= "same")
#2eme conv
conv_model <- layer_conv_2d(conv_model, filters=256, kernel_size = 5, strides = 1, padding = "same", activation = "relu")
conv_model <- layer_max_pooling_2d(conv_model, pool_size = 2, padding= "same")
#3eme conv
conv_model <- layer_conv_2d(conv_model, filters=384, kernel_size = 4, strides = 1, padding = "same", activation = "relu")
conv_model <- layer_conv_2d(conv_model, filters=384, kernel_size = 4, strides = 1, padding = "same", activation = "relu")
conv_model <- layer_conv_2d(conv_model, filters=256, kernel_size = 3, strides = 1, padding = "same", activation = "relu")
conv_model <- layer_max_pooling_2d(conv_model, pool_size = 2, padding= "same", strides=2)
#fully connected
conv_model <- layer_flatten(conv_model)
conv_model <- layer_dense(conv_model, units = 250, activation = "relu")
conv_model <- layer_dropout(conv_model, 0.2)
conv_model <- layer_dense(conv_model, units = 150, activation = "relu")
conv_model <- layer_dropout(conv_model, 0.2)
conv_model <- layer_dense(conv_model, units = 100, activation = "relu")
conv_model <- layer_dropout(conv_model, 0.2)
conv_model <- layer_dense(conv_model, units = 10, activation = "softmax")

compile(conv_model, loss = "categorical_crossentropy", optimizer = "adam", metrics= "accuracy")
fit(conv_model, x_train[1:1000,,,], y_train[1:1000,], epochs = 100, batch_size = 10,validation_split = 0.3)

evaluate(object = conv_model, x_train, y_train)
evaluate(object=conv_model, x_test, y_test)
