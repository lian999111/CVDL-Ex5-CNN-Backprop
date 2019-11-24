# %%
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import DLCVDatasets

# %% Prepare CIFAR-10 dataset
train_size = 60000
test_size = 10000
used_labels = list(range(0, 10))    # the labels to be loaded
x_train, y_train, x_test, y_test, class_names = DLCVDatasets.get_dataset('cifar10',
                                                                         used_labels=used_labels,
                                                                         training_size=train_size,
                                                                         test_size=test_size)
# Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0
input_shape = x_train.shape[1:4]
# %% Define fully connected model
fully_connected_model = k.Sequential([
    k.layers.Flatten(input_shape=input_shape),
    k.layers.Dense(128, activation=tf.nn.relu),
    k.layers.Dense(64, activation=tf.nn.relu),
    k.layers.Dense(64, activation=tf.nn.relu),
    k.layers.Dense(10, activation=tf.nn.softmax)
])

fully_connected_model.summary()

# %% Callback for early stopping
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.9):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True

# %% Compile fully_connected model
from tensorflow.keras.optimizers import Adam
fully_connected_model.compile(optimizer=Adam(lr=0.001),
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])


# %% Train model
fully_connected_model.fit(x_train, y_train, validation_data=(x_test, y_test),
                          epochs=15, batch_size=64, verbose=2)

# %%
