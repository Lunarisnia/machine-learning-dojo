import cv2 as cv
import numpy as np
from keras import models
import pyttsx3 as pytx

CLASS_NAMES = ['C', 'D', 'E', 'F', 'H', 'K', 'L', 'O', 'R', ' ', 'U', 'W']
cap = cv.VideoCapture(0)
sign_model = models.load_model('./savedModels/sign_asl_personal02')

predictedWord = ''
textBar = np.ones((150, 700))

def nothing(x):
    pass

cv.namedWindow('Frame')
cv.createTrackbar('minVal', 'Frame', 0, 255, nothing)
cv.createTrackbar('maxVal', 'Frame', 0, 255, nothing)
while 1:
    _, frame = cap.read()
    frame = frame[200:480, 200:640]
    resized = cv.resize(frame, (64, 64))

    cv.putText(textBar, predictedWord, (30, 100), color=(0, 255, 0), thickness=3, lineType=cv.LINE_AA, fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=2)
    cv.imshow('textBar', textBar)
    # cv.imshow('Frame', frame)
    
    minVal = cv.getTrackbarPos('minVal', 'Frame')
    maxVal = cv.getTrackbarPos('maxVal', 'Frame')
    # ret, mask = cv.threshold(frame, 105, 228, cv.THRESH_BINARY)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(gray, minVal, maxVal, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    test = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('Frame', frame)

    key_pressed = cv.waitKey(1)
    if key_pressed == ord('q'):
        break
    elif key_pressed == ord('s'):
        feature = np.array([resized])
        raw_prediction = sign_model.predict(feature)
        prediction = np.argmax(raw_prediction.flatten())
        print(CLASS_NAMES[prediction])
        predictedWord = predictedWord + CLASS_NAMES[prediction]
    elif key_pressed == ord('r'):
        predictedWord = ''
        textBar = np.ones((150, 700))
        print("Reset!")
    elif key_pressed == ord('p'):
        print(f"You Said: {predictedWord}")
        engine = pytx.init()
        engine.say(predictedWord)
        engine.runAndWait()

cap.release()
cv.destroyAllWindows()