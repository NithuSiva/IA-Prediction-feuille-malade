import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from google.colab import drive

# drive.mount('/content/drive')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# train_path = "/content/drive/MyDrive/projet_IA/train"
# test_path = "/content/drive/MyDrive/projet_IA/test"

train_dir = pathlib.Path('data/train')
test_dir = pathlib.Path('data/test')

def dataSet(validation_split, img_size, batch_size, type, dir):
    data_set = tf.keras.utils.image_dataset_from_directory(
                dir,
                validation_split=validation_split,
                subset=type,
                seed=123,
                image_size=(img_size, img_size),
                batch_size=batch_size)
    return data_set

def getClassNames(data_set):
    return data_set.class_names


def createModel(num_classes, nb_layers, hidden_activation, output_activation, hidden_size, output_size, img_size):
    
    model = keras.Sequential()
    model.add(layers.Rescaling(1./255, input_shape=(img_size, img_size, 3)))
    
    for nb_layer in range(nb_layers):
        model.add(layers.Conv2D(hidden_size, 3, padding='same', activation=hidden_activation))
        model.add(layers.MaxPooling2D())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(output_size, activation=output_activation))
    model.add(layers.Dense(num_classes))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model


def trainModel(model, epochs, train_ds, val_ds):
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs)

def main():
    batch_size = 32
    img_height = 400
    img_width = 400

    train_ds = dataSet(0.2, img_height, batch_size, 'training', train_dir)
    val_ds = dataSet(0.2, img_height, batch_size, 'validation', train_dir)

    class_names = getClassNames(train_ds)

    model = createModel(len(class_names), 3, 'relu', 'softmax', 64, 128, img_height)

    trained_model = trainModel(model, 20, train_ds, val_ds)


    trained_model.save("/model/model_data_augment.h5")

    return 0

main()