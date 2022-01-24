import cv2
import numpy as np

###################### ОБЪЯВЛЕНИЕ ПЕРЕМЕННЫХ ######################
cap = cv2.VideoCapture(0)

# параметры детектируемых маркеров
arDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
arPars = cv2.aruco.DetectorParameters_create()

# параметры ПД регулятора для выравниания по маркерам
k1 = 1
k2 = 1
err = 0
errold = 0
u = 0

# ширина и высота изображения
W, H = (640, 480)
# флаг необходимости изменять размер изображения под установленный выше
flag_resize = False
###################################################################


######################## ОБЪЯВЛЕНИЕ ФУНКЦИЙ #######################
def get_aruco_center(corners):
    '''
    Calculates marker center coordinates from its corners
    :param corners: array with 4 corners
    :return: array with center coordinates
    '''
    x = np.sum([ c[0] for c in corners[0] ]) // 4
    y = np.sum([ c[1] for c in corners[0] ]) // 4
    return np.int(x), np.int(y)
###################################################################


############################### MAIN ##############################
_, frame = cap.read()
if len(frame) != H:
    flag_resize = True

while True:
    _, frame = cap.read()
    if flag_resize:
        frame = cv2.resize(frame, (640, 480))

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