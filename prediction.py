import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import cv2

train_path = "data/train"
test_path = "data/test"

train_dir = pathlib.Path(train_path)
test_dir = pathlib.Path(test_path)
 

def afficher_folder(img_folder):
    image_files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]

    num_images_to_display = min(9, len(image_files))

    plt.figure(figsize=(10, 10))

    for i in range(num_images_to_display):
        img_path = os.path.join(img_folder, image_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        plt.axis('off')

        plt.title(image_files[i], fontsize=8, pad=2)

    plt.show()

def predict_folder(model, img_test_dir, img_folder, img_size, class_names):
    img_list = []
    for img in img_test_dir:
        img_path = img_folder + '/' + img
        img_loaded = tf.keras.utils.load_img(
            img_path, target_size=(img_size, img_size)
        )

        img_array = tf.keras.utils.img_to_array(img_loaded)
        img_array = tf.expand_dims(img_array, 0) 
        img_list.append([img, img_array])

    i = 1
    for img_name, img_array in img_list:
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        img_path = img_folder + '/' + img_name
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        plt.subplot(3, 3, i)
        i = i + 1
        plt.imshow(img)
        plt.axis('off')
        predict_string = class_names[np.argmax(score)] + " " +  str(round(100 * np.max(score))) + "%"
        plt.title(predict_string, fontsize=8, pad=2)
        
    plt.show()

def trainingDataSet(batch_size, img_size):
    print('')
    print("* -----------TRAINING DATA SET-------------- *")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    print('')
    print("* ----------FIN TRAINING DATA SET-------------- *")
    return train_ds

def validationDataSet(batch_size, img_size):
    print('')
    print("* -----------VALIDATION DATA SET-------------- *")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    print('')
    print("* ----------FIN VALIDATION DATA SET-------------- *")
    return val_ds

def getClassNames(dataset):
    return dataset.class_names

def loadModel(model_path):
    return tf.keras.models.load_model(model_path)

def info():
    print('')
    print("* -----------INFO-------------- *")
    print("Python 3.9.0 / prediction.py")
    image_count = len(list(train_dir.glob('*/*.jpg')))
    print('')
    print("Nombre d'images dans le data/train: ", image_count)
    print('')
    image_count = len(list(test_dir.glob('*/*.jpg')))
    print('')
    print("Nombre d'images dans le data/test: ", image_count)
    print('')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("* -----------FIN INFO-------------- *")
    print('')

def main():
    info()
    batch_size = 32
    projet_img_size = 400
    kaggle_img_size = 224

    # Entraine
    train_ds = trainingDataSet(batch_size, projet_img_size)
    val_ds = trainingDataSet(batch_size, projet_img_size)

    model_projet = loadModel('model/model_data_augment.h5')
    model_kaggle = loadModel('model/my_keras_model.h5')
    class_name = getClassNames(train_ds)

    # PREDICTION 1
    img_folder = "data/new_leaf/Apple Rust Leaf"

    img_test_dir = os.listdir(img_folder)
    afficher_folder(img_folder)
    predict_folder(model_projet, img_test_dir, img_folder, projet_img_size, class_name)


    # PREDICTION 2
    img_folder = "data/new_leaf/Tomato Early Blight"

    img_test_dir = os.listdir(img_folder)
    # afficher_folder(img_folder)
    predict_folder(model_projet, img_test_dir, img_folder, projet_img_size, class_name)
    predict_folder(model_kaggle, img_test_dir, img_folder, kaggle_img_size, class_name)
    
main()