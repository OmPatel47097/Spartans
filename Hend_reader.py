# Collecting the data of hand posters.

import cv2
import os
import sys
import time


#lable = sys.argv[1]
sample_imj = 200

IMG_s_PATH = 'D:\ML'
IMG_cls_PATH = 'D:\ML\TheOne'

capt = cv2.VideoCapture(0)

start = True
count = 0


while True:
    ret, frame = capt.read()
    if not ret:
        continue

    if count == sample_imj:
        break

    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    if start:
        roi = frame[100:500, 100:500]
        save_path = os.path.join(IMG_cls_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(50)
    if k == ord('a'):
        start = not start

    if k == ord('q'):
        break

print("\n{} image(s) saved to {}".format(count, IMG_cls_PATH))
capt.release()
cv2.destroyAllWindows()
