from keras.models import load_model
import cv2
import numpy as np
from random import choice

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


def calcu_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "Rock":
        if move2 == "Scissors":
            return "User"
        if move2 == "Paper":
            return "Computer"
        if move2 == "Spock":
            return "Computer"
        if move2 == "Lizard":
            return "User"

    if move1 == "Paper":
        if move2 == "Rock":
            return "User"
        if move2 == "Scissors":
            return "Computer"
        if move2 == "Spock":
            return "User"
        if move2 == "Lizard":
            return "Computer"


    if move1 == "Scissors":
        if move2 == "Paper":
            return "User"
        if move2 == "Rock":
            return "Computer"
        if move2 == "Lizard":
            return "User"
        if move2 == "Spock":
            return "Computer"

    if move1 == "Lizard":
        if move2 == "Paper":
            return "User"
        if move2 == "Rock":
            return "Computer"
        if move2 == "Scissors":
            return "Computer"
        if move2 == "Spock":
            return "User"

    if move1 == "Spock":
        if move2 == "Paper":
            return "Computer"
        if move2 == "Rock":
            return "User"
        if move2 == "Scissors":
            return "User"
        if move2 == "Lizard":
            return "Computer"

model = load_model("rock-paper-scissors-lizard-spock-FIFTY.h5")

cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['Rock', 'Paper', 'Scissors', 'Lizard', 'Spock'])
            winner = calcu_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread(
            "Hands/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (400,400))
        frame[400:400, 800:800] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
