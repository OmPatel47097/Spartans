from keras.models import load_model
import cv2
import numpy as np
import sys

a = input("Enter Path:")
filepath = a

REV_CLASS_MAP = {
    0: "Rock",
    1: "Paper",
    2: "Scissors",
    3: "Lizard",
    4: "Spock",
    5: "Blank",
    6: "TheOne"
}


def mapper(val):
    return REV_CLASS_MAP[val]


model = load_model("rock-paper-scissors-model.h5")

# load the image
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (227, 227))

# predict the move
pred = model.predict(np.array([img]))
# print(pred)
move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))
