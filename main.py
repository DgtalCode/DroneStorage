import cv2
import numpy as np

cap = cv2.VideoCapture(0)

arDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
arPars = cv2.aruco.DetectorParameters_create()

k1 = 1
k2 = 1
err = 0
errold = 0
u = 0

# marker = np.zeros((500,500), dtype=np.uint8)
# marker = cv2.aruco.drawMarker(arDict, 0, 500, marker, 1)
# cv2.imwrite('aruco_6x6_250_0.png', marker)

def get_aruco_center(corners):
    x = np.sum([ c[0] for c in corners[0] ]) // 4
    y = np.sum([ c[1] for c in corners[0] ]) // 4
    return np.int(x), np.int(y)

while True:
    ret, frame = cap.read()

    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arDict, parameters=arPars)

    if corners:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids, (255,0,0))

    for c in corners:
        x, y = get_aruco_center(c)
        cv2.drawMarker(frame, (x,y), (0,0,255), cv2.MARKER_CROSS)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()