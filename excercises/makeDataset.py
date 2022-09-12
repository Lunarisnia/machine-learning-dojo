import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
toggle = 0
i = 0

while 1:
    if toggle:
        cv.imwrite(f'personalDatasets/asl_personal/train/D/D{i}.jpg', frame)
        i = i + 1
        if i % 100 == 0: print(f"Progress: {i}")
        if i == 2999: break
    _, frame = cap.read()

    frame = frame[200:480, 200:640]
    cv.imshow('Frame', frame)

    waitKey = cv.waitKey(1)
    if waitKey == ord('q'):
        break
    elif waitKey == ord('s'):
        toggle = 1
        # cv.imwrite(f'personalDatasets/asl_personal/train/A/A{i}.jpg', frame)
        # print(f"Saved! {i}")
        # i = i + 1

cap.release()
cv.destroyAllWindows()
