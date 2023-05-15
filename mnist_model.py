import tensorflow as tf
import numpy as np


(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.reshape(x_train, (60000, 28*28))
x_test = np.reshape(x_test, (10000, 28*28))

x_train = x_train/255.0
x_test = x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)

_ = model.fit(
    x_train,y_train,
    validation_data=(x_test,y_test),
    epochs = 20,batch_size = 1024
)

model.save('model.h5')
