import cv2
import numpy as np
from pioneer_sdk import Pioneer

###################### ОБЪЯВЛЕНИЕ ПЕРЕМЕННЫХ ######################
# ыбор источника видео:
# 0 - камера компьютера
# 1 - камера квадрокоптера
flag_video_source = 0

if flag_video_source == 1:
    pioneer = Pioneer(logger=False, bad_connection_exit=False)
    if pioneer.bad_connection_occured():
        flag_video_source = 0

if flag_video_source == 0:
    cap = cv2.VideoCapture(0)


# параметры детектируемых маркеров
arDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
arPars = cv2.aruco.DetectorParameters_create()

# параметры ПД регулятора для выравниания по маркерам
k1 = 0.0008
k2 = 0.002
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
    return int(x), int(y)


def vec_from_points(point_start, point_end):
    return (point_end[0] - point_start[0],
            point_start[1] - point_end[1])


def vec_length(vec):
    return np.sqrt( vec[0]**2 + vec[1]**2 )


def vec_direction(vec, to_degrees=False):
    angle = np.arctan2(vec[1], vec[0])
    if to_degrees:
        angle = np.degrees(angle)
    return angle


def distance_between_points(point1, point2):
    return np.sqrt( (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 )
###################################################################


############################### MAIN ##############################
_, frame = cap.read()
if len(frame) != H:
    flag_resize = True

while True:
    # считывание изображения
    _, frame = cap.read()
    if flag_resize:
        frame = cv2.resize(frame, (640, 480))

    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arDict, parameters=arPars)
    if corners:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids, (255,0,0))

    if len(corners) == 1:
        x, y = get_aruco_center(corners[0])
        cv2.drawMarker(frame, (x,y), (0,0,255), cv2.MARKER_CROSS)

        error_vec = vec_from_points((W // 2, H // 2), (x, y))
        error_vec_dir = vec_direction(error_vec)
        cv2.arrowedLine(frame, (W // 2, H // 2), (x, y), (255, 0, 255), 2)

        err = vec_length(error_vec)
        u = k1*err - k2*(err-errold)
        errold = err

        x_correction = round(u * np.cos(error_vec_dir), 4)
        y_correction = round(u * np.sin(error_vec_dir), 4)

        print(x_correction, y_correction)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()