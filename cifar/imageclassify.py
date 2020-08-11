import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def visualizeData(train_generator):
    labels = (train_generator.class_indices)
    print(labels)
    class_names = [k for k,v in labels.items()]

    print(class_names)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,10))
    for images, classes in train_generator:
        print("batch shape=%s, min=%.3f, max=%.3f" %(images.shape, images.min(), images.max()))
        #print("batch labels shape=%s" %(labels.shape))
        #print(labels)
        for i in range(9):
            ax = plt.subplot(3,3,i+1)
            plt.imshow(images[i])
            class_index = classes[i].tolist().index(1)
            plt.title(class_names[class_index])
            plt.axis("on")

        plt.show()


print(tf.__version__)

import pathlib

def get_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir_str = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    print("data_dir: " + data_dir_str)
    data_dir = pathlib.Path(data_dir_str)
    print(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    return data_dir_str

def show_image(data_dir_str):
    data_dir = pathlib.Path(data_dir_str)
    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0]))
    tulips = list(data_dir.glob('tulips/*'))
    PIL.Image.open(str(tulips[0])).show()

batch_size = 32
img_height = 180
img_width = 180
IMAGE_SIZE = (img_height, img_width)
data_dir_str = get_dataset()

def get_train_generator_0():
    datagen = ImageDataGenerator(rescale=1./255)
    train_it = datagen.flow_from_directory(
        data_dir_str,
        target_size=(img_height, img_width),
        class_mode='categorical'
    )
    return train_it

def get_train_generator():
    datagen_kwargs = dict(rescale=1./255, validation_split=.20)
    dataflow_kwargs = dict(target_size=(img_height, img_width), batch_size=batch_size,
                            interpolation="bilinear")
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs) 
    valid_generator = valid_datagen.flow_from_directory(data_dir_str, subset="validation", 
                        shuffle=False, **dataflow_kwargs)
    do_data_augmentation = True
    if do_data_augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            **datagen_kwargs
        )
    else:
        train_datagen = valid_datagen

    train_generator = train_datagen.flow_from_directory(
        data_dir_str, subset="training", shuffle=True, **dataflow_kwargs
    )
    images, labels = next(train_generator)

    print(images.dtype, images.shape)
    print(labels.dtype, labels.shape)
    return (train_generator, valid_generator)


#visualizeData(get_train_generator_0())
#visualizeData(get_train_generator()[0])

def get_dataset(data_generator):
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator,
        output_types=(tf.float32, tf.float32)
        #,output_shapes=([32, img_height, img_width, 3], [32, 5])
        ,output_shapes=([None, img_height, img_width, 3], [None, 5])
    )
    ds = ds.prefetch(32)
    return ds

def get_model():
    num_classes = 5
    model = Sequential([
        layers.Conv2D(32,3, activation='relu', input_shape=(img_height,img_width,3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.summary()                
    return model

def get_model_v2():
    num_classes = 5
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height,img_width,3)),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['acc'])
    model.summary()
    return model

def train_model(model):
    data_generator = get_train_generator()
    history = model.fit_generator(
        data_generator[0],
        steps_per_epoch=500,
        epochs=10,
        validation_data=data_generator[1],
        validation_steps=50
    )
    return history

def draw_history_graph(history):
    acc_train = history.history['acc']
    acc_val = history.history['val_acc']
    epochs = range(1,81)
    plt.plot(epochs, acc_train, 'g', label='training accuracy')
    plt.plot(epochs, acc_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

draw_history_graph(train_model(get_model_v2()))
