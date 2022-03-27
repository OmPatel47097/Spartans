import cv2
import numpy as np
import tensorflow as tf
from keras_squeezenet import SqueezeNet
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import os

IMG_SAVE_PATH = 'S:/Compitation & Hackthons/not_Main/ML'

Cls_map = {
    "Rock": 0,
    "Paper": 1,
    "Scissors": 2,
    "Lizard": 3,
    "Spock": 4,
    "Blank": 5,
    "TheOne": 6
}

NUM_CLASSES = len(Cls_map)


def mapper(val):
    return Cls_map[val]


def gg_model():
    model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model


# load images from the Privous data wich we have collected.
dataset = []
for directory in os.listdir('S:\Compitation & Hackthons\Sparten_Try\ML'):
    path = os.path.join('S:\Compitation & Hackthons\Sparten_Try\ML', directory)

    for item in os.listdir(path):
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

'''
dataset = [
    [[...], 'rock'],
    [[...], 'paper']
]
'''
data, labels = zip(*dataset)
labels = list(map(mapper, labels))

'''
labels: rock,paper,paper,scissors,rock...
one hot encoded: [1,0,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]...
'''

# one hot encode the labels
labels = np_utils.to_categorical(labels)

# define the model
model = gg_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# start training
model.fit(np.array(data), np.array(labels), epochs=50)

# save the model for later use
model.save("rock-paper-scissors-lizard-spock-FIFTY.h5")
