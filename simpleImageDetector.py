#A simple hand ASL-style detector that reads in 6 images and senses based on hand tracking the type (using opencv
#and mediapipe; very sensitive to motion); Code authored by @moshah122

import cv2 as cv
import mediapipe as mp
import time

#sets the webcam environment and initializes the features of mediapipe
cam = cv.VideoCapture(0)
widthCam = 1080
heightCam = 720
cam.set(3, widthCam)
cam.set(4, heightCam)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#initializes time variables and creates a list of related captions for each of the ASL images
timeBegan = False
beginTime = 0
compString = ""
wordList = []
captionList = ["hi", "love", "peace", "thumbs down", "thumbs up", "walk"]

#while the webcam is reading...
while True:
    #reads in the webcam and returns the frame image and sets up a list to store all 21 landmarks
    ret, img = cam.read()
    height, width, x = img.shape
    rgbImg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    res = hands.process(rgbImg)
    landmarkList = []
    sign = ""
    minX = 2000
    maxX = 0
    minY = 2000
    maxY = 0
    #if there are hands detected in the frame, it will draw the landmarks and find the boundaries of x and y coordinates
    if (res.multi_hand_landmarks):
        for mulHands in res.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, mulHands, mpHands.HAND_CONNECTIONS)
            hand = res.multi_hand_landmarks[0]
            for lmId, mark in enumerate(hand.landmark):
                centerX = int(mark.x * width)
                centerY = int(mark.y * height)
                minX = min(centerX, minX)
                maxX = max(centerX, maxX)
                minY = min(centerY, minY)
                maxY = max(centerY, maxY)
                landmarkList.append([centerX, centerY])
    #if the frame is not empty...
    if len(landmarkList) != 0:
        #for hi and love signs respectively
        if ((landmarkList[4][1] <= landmarkList[3][1]) and (landmarkList[8][1] <= landmarkList[7][1]) and
                (landmarkList[20][1] <= landmarkList[19][1])):
            if (landmarkList[12][1] <= landmarkList[11][1]) and (landmarkList[16][1] <= landmarkList[15][1]):
                sign = captionList[0]
            elif (landmarkList[12][1] >= landmarkList[11][1]) and (landmarkList[16][1] >= landmarkList[15][1]):
                sign = captionList[1]
            else:
                sign = "Not Detected"
        #for thumbs up and thumbs down respectively
        elif ((landmarkList[8][0] <= landmarkList[7][0]) and (landmarkList[12][0] <= landmarkList[11][0]) and
                (landmarkList[16][0] <= landmarkList[15][0]) and (landmarkList[20][0] <= landmarkList[19][0])):
            if landmarkList[4][1] <= landmarkList[3][1]:
                sign = captionList[4]
            elif landmarkList[4][1] >= landmarkList[3][1]:
                sign = captionList[3]
            else:
                sign = "Not Detected"
        #for peace and walk respectively
        elif ((landmarkList[20][1] >= landmarkList[19][1]) and (landmarkList[16][1] >= landmarkList[15][1]) and
                (landmarkList[8][1] <= landmarkList[7][1]) and (landmarkList[12][1] <= landmarkList[11][1]) and
              ((landmarkList[4][0] <= landmarkList[16][0]))):
            sign = captionList[2]
        elif ((landmarkList[20][1] <= landmarkList[19][1]) and (landmarkList[16][1] <= landmarkList[15][1]) and
                (landmarkList[8][1] >= landmarkList[7][1]) and (landmarkList[12][1] >= landmarkList[11][1]) and
              ((landmarkList[4][0] >= landmarkList[16][0]))):
            sign = captionList[5]
        #else, return that there is no detection
        else:
            sign = "Not Detected"
    #if sign detection is true...
    if sign != "Not Detected" and sign != "":
        #start the time if a sign is detected
        if timeBegan == False:
            beginTime = time.time()
            compString = sign
            timeBegan = True
        else:
            #if the current sign is not equal to the previous sign, restart the timer
            if sign != compString:
                timeBegan = False
            #if the total time of the hand sign is greater than 1.5 seconds, add it to the recurring word list
            elif time.time() - beginTime >= 1.5:
                wordList.append(compString)
                timeBegan = False
                cv.rectangle(img, (minX, minY), (maxX, maxY), (0, 255, 0), 6)
    #else, there is no such detection
    else:
        timeBegan = False
    cv.putText(img, sign, (700, 575), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)
    #keeps record of the word calls 5 at a time (uses the word list as a way to store the words)
    cv.putText(img, "List of Calls", (50, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    if len(wordList) > 5:
        str = wordList[5]
        wordList.clear()
        wordList.append(str)
    for i in range(0, len(wordList)):
        cv.putText(img, wordList[i], (50, 90 + 40 * i), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.imshow("Image", img)
    #if the 'd' key is pressed, close the program
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

#release the webcam and quit the frames
cam.release()
cv.destroyAllWindows()