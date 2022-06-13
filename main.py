import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


image = cv2.imread('CVtask.jpg')
imgGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow("img", image)



Arucos = [0,0,0,0]
Arucos[0] = cv2.imread('Ha.jpg')
Arucos[1] = cv2.imread('HaHa.jpg')
Arucos[2] = cv2.imread('XD.jpg')
Arucos[3] = cv2.imread('LMAO.jpg')

for i in range(4):
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    arucoParam = cv2.aruco.DetectorParameters_create()
    imgGray = cv2.cvtColor(Arucos[i], cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(Arucos[i], arucoDict, parameters=arucoParam)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(image, [approx], 0, (0, 0, 0), 5)
    a = approx.ravel()[0]
    b = approx.ravel()[1] - 5
    if len(approx) == 4:
        a1, b1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        asp1 = float(a1) / w
        asp2 = float(b1) / h
        #print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            cv2.putText(image, "square", (a, b), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
            cv2.rectangle(image, (a1, b1), (a1 + w, b1 + w), (0, 255, 0), 10)
            x2 = a1 + w
            y2 = b1 + h
cv2.imshow("img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#there is not a final.png file in the repo sir because i was not able to find the cordinates of the corner to rotate and paste the image sorry sir
