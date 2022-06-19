import numpy as np
import cv2
import cv2.aruco as aruco
import imutils
import math

Aruco1 = (cv2.imread("Ha.jpg"))
Aruco2 = (cv2.imread("HaHa.jpg"))
Aruco3 = (cv2.imread("LMAO.jpg"))
Aruco4 = (cv2.imread("XD.jpg"))

Aruco_list = (Aruco1, Aruco2, Aruco3, Aruco4)


def findAruco(Img):
    Gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_5X5_250')
    ArucoDict = aruco.Dictionary_get(key)
    ArucoPara = aruco.DetectorParameters_create()
    (Corners, Ids, Rejected) = aruco.detectMarkers(Gray, ArucoDict, parameters = ArucoPara)
    return (Corners, Ids, Rejected)

id_list = []
for i in Aruco_list:
    id_list.append((findAruco(i))[1][0][0])

def color(Color, ll, ul):
    if Color[0] in range(ll[0], ul[0] + 1):
        if Color[1] in range(ll[1], ul[1] + 1):
            if Color[2] in range(ll[2], ul[2] + 1):
                return True
    else:
        return False

def aruco_Coords(Img, Aru_Length, Aru_Height, Bound_Height, Bound_Length, angle):
    (c, a, b) = findAruco(Img)
    if len(c)>0:
        a=a.flatten()
        for (markercorner,markerid) in zip(c, a):
            Corner = markercorner.reshape((4, 2))
            (topleft,topright,bottomright,bottomleft) = Corner
            bottomleft = (int(bottomleft[0]),int(bottomleft[1]))
            bottomright = (int(bottomright[0]),int(bottomright[1]))
            m = ((bottomright[1]-bottomleft[1])/(bottomright[0]-bottomleft[0]))
            fi = math.atan(m)
            a = fi * 180/math.pi
            Img = imutils.rotate_bound(Img, -a)
            (c, a, b) = findAruco(Img)
            if len(c) > 0:
                a = a.flatten()
                for (markercorner, markerid) in zip(c, a):
                    Corner = markercorner.reshape((4, 2))
                    (topleft, topright, bottomright, bottomleft) = Corner


                    topleft = (int(topleft[0]), int(topleft[1]))
                    topright = (int(topright[0]), int(topright[1]))
                    bottomright = (int(bottomright[0]), int(bottomright[1]))
                    bottomleft = (int(bottomleft[0]), int(bottomleft[1]))
                    Img = Img[topleft[1]:bottomright[1], topleft[0]:bottomright[0]]
                    blank=np.zeros((int(Bound_Height), int(Bound_Length), 3))
            s=np.shape(blank[int((int(Bound_Height) - int(Aru_Height)) / 2):int((int(Bound_Length) + int(Aru_Length)) / 2), int((int(Bound_Length) - int(Aru_Length)) / 2):int((int(Bound_Length) + int(Aru_Length)) / 2)])

            Img=cv2.resize(Img, (s[1], s[0]))
            Img=blank[int((int(Bound_Height) - int(Aru_Height)) / 2):int((int(Bound_Length) + int(Aru_Length)) / 2), int((int(Bound_Length) - int(Aru_Length)) / 2):int((int(Bound_Length) + int(Aru_Length)) / 2)]
            Img=imutils.rotate(blank, angle)
            return Img

Img = cv2.imread("CVtask.jpg")
def final_func(Img, id_list):
        Img = cv2.imread("CVtask.jpg")
        Gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        ret , thresh = cv2.threshold(Gray, 230, 255, cv2.THRESH_BINARY)
        cont,heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for cnt in cont:
            Approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(Approx)

            if len(Approx) == 4:
                if float(w)/h >= 0.95 and float(w)/h <= 1.05:

                    Slope = (Approx[1][0][1] - Approx[2][0][1]) / (Approx[1][0][0] - Approx[2][0][0])
                    Angle = 180 / math.pi * math.atan(Slope)
                    dx = math.sqrt((Approx[1][0][1] - Approx[2][0][1]) ** 2 + (Approx[1][0][0] - Approx[2][0][0]) ** 2)
                    dx = int(dx)
                    dy = math.sqrt((Approx[2][0][1] - Approx[3][0][1]) ** 2 + (Approx[2][0][0] - Approx[3][0][0]) ** 2)
                    dy = int(dy)

                    if color(Img[int(y + (h / 2)) , int(x + (w / 2))], (0, 128, 0), (152, 255, 154)):
                        ind = id_list.index(1)
                        Aruco_Img = aruco_Coords(Aruco_list[ind], dx, dy, h, w, -Angle)
                        print(1)
                    elif color(Img[int(y + (h / 2)), int(x + (w / 2))], (0, 100, 200), (153, 204, 255)):
                        ind = id_list.index(2)
                        Aruco_Img = aruco_Coords(Aruco_list[ind], dx, dy, h, w, -Angle)
                        print(2)
                    elif color(Img[int(y + (h / 2)), int(x + (w / 2))], (0, 0, 0), (20, 20, 20)):
                        ind = id_list.index(3)
                        Aruco_Img = aruco_Coords(Aruco_list[ind], dx, dy, h, w, -Angle)
                        print(3)
                    elif color(Img[int(y + (h / 2)), int(x + (w / 2))], (200, 200, 200), (250, 250, 250)):
                        ind = id_list.index(4)
                        Aruco_Img = aruco_Coords(Aruco_list[ind], dx, dy, h, w, -Angle)
                        print(4)
                    cv2.drawContours(Img, [Approx], -1, (0, 0, 0), -1)
                    Img[y:y + h, x:x + w] = Img[y:y + h, x:x + w] + Aruco_Img
        return Img

final_func(Img, id_list)

cv2.namedWindow("FINAL_IMAGE",cv2.WINDOW_NORMAL)
cv2.imshow("FINAL_IMAGE", final_func(Img, id_list))
cv2.waitKey(0)
cv2.destroyAllWindows()
