import cv2 as cv
# using the handdetector from cvzone package which relies on mediapipe library
from cvzone.HandTrackingModule import HandDetector

# 0 is for the default webcam
cap = cv.VideoCapture(0)

# this is for increasing the size of the screen
# using HD resolution
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8) # detection confidence

while True:
    success, img = cap.read()
    # finding the hand 
    lmList, bboxInfo = detector.findHands(img,True)
    cv.imshow("Image", img)
        
    # line to terminate the program
    # press 'q' to terminate the program
    if cv.waitKey(2) & 0xFF == ord('q'):
        break    
