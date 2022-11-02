import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
toggle = 0
i = 2999


def nothing(x):
    pass

cv.namedWindow('Frame')
cv.createTrackbar('minVal', 'Frame', 0, 255, nothing)
cv.createTrackbar('maxVal', 'Frame', 0, 255, nothing)

while 1:
    if toggle:
        cv.imwrite(f'personalDatasets/asl_personal/train/D/D{i}.jpg', frame)
        i = i + 1
        if i % 100 == 0: print(f"Progress: {i}")
        if i == 2999 + 2999: break
    _, frame = cap.read()

    frame = frame[200:480, 200:640]
    # cv.imshow('Frame', frame)
    minVal = cv.getTrackbarPos('minVal', 'Frame')
    maxVal = cv.getTrackbarPos('maxVal', 'Frame')
    # ret, mask = cv.threshold(frame, 105, 228, cv.THRESH_BINARY)
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # ret, mask = cv.threshold(gray, minVal, maxVal, cv.THRESH_BINARY)
    # mask_inv = cv.bitwise_not(mask)
    # test = cv.bitwise_and(frame, frame, mask=mask)
    # cv.imshow('Frame', cv.bitwise_not(mask_inv))
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
