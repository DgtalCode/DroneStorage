import random
import sys
import cv2
import numpy as np
from pioneer_sdk.asynchronous import Pioneer
from pioneer_sdk import pioutils

###################### ОБЪЯВЛЕНИЕ ПЕРЕМЕННЫХ ######################

# объект для взаимодействия с квадрокоптером
pioneer = Pioneer(logger=False)

# параметры детектируемых маркеров
arDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)  # тип маркера
arPars = cv2.aruco.DetectorParameters_create()  # параметры детектирования (стандартные)

# массив, хранящий найденные маркеры по позициям
database = []

# позиционировнаие коптера
mvxy = -1  # шаг перемещений по Х У (м)
mvz = 1.7  # шаг перемещений по Z (м)
mvyaw = np.radians(90)  # шаг поворота (град)

# параметры ПД регулятора для выравниания по маркерам
k1 = 0.12  # реакция на отклонение
k2 = 0.09  # смягчение резких движений
k3 = 0.0004 # реакция на маленькие отклонения (сумму ошибки)
err = 0  # ошибка (величина отклонения)
errold = 0  # старая ошибка (величина отклонения на предыдущей итерации)
errsum = 0 # сумма ошибок
u = 0  # управляющее воздействие

# ширина и высота изображения
W, H = (640, 480)

###################################################################


######################## ОБЪЯВЛЕНИЕ ФУНКЦИЙ #######################
def get_aruco_center(corns):
    """
    Считает координаты центра маркера по переданнм четырем углам
    :param corns: массив с коодинатами углов маркера
    :return: массив с координатами центра маркера
    """
    x = np.sum([c[0] for c in corns]) // 4
    y = np.sum([c[1] for c in corns]) // 4
    return int(x), int(y)


def nothing(x):
    pass


def add_marker(name, data_raw):
    """
    Добавляет информацию о считанном маркере в базу данных (массив)
    :return:
    """
    global database
    data = None
    if data_raw is not None:
        data_raw = data_raw[1][0]
        print(data_raw)
        data = {
            name: {
                'type': data_raw & 0b111110000,
                'subtype': data_raw & 0b000001111
            }
        }
    else:
        data = {
            name: 'empty'
        }
    database.append(data)
###################################################################


############################### MAIN ##############################
rows = 1
cols = 2
row_cur = 1
col_cur = 1
while True:
    # считывание изображения
    frame = cv2.imdecode(np.frombuffer(pioneer.get_raw_video_frame(), dtype=np.uint8), cv2.IMREAD_COLOR)

    # детектирование маркеров
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arDict, parameters=arPars)

    if corners:
        markers_data = list(enumerate(corners))
        markers_data = sorted(markers_data, key=lambda md: pioutils.distance_between_points(
            get_aruco_center(md[1][0]),
            (W//2, H//2)
        ), reverse=True)
        markers_data = (markers_data[0][1], ids[markers_data[0][0]])
    else:
        markers_data = None

    # отображение маркеров на изображении
    if corners:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids, (255, 0, 0))

    if pioneer.step_get() == 0:
        pioneer.arm()
        pioneer.takeoff()
        if pioneer.in_air():
            pioneer.step_inc()

    if pioneer.step_get() == 1:
        pioneer.go_to_local_point(x=0, y=0, z=mvz * row_cur, callback=lambda: pioneer.step_inc(2))

    if pioneer.step_get() == 2:
        u, err, error_vec_dir = 0, 0, 0

        if corners:
            # расчет центра маркера
            x, y = get_aruco_center(corners[0])
            cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS)

            # расчет отклонения маркера от центра изображения в виде вектора
            error_vec = pioutils.vec_from_points((W // 2, H // 2), (x, y))
            error_vec_dir = round(pioutils.vec_direction(error_vec, to_degrees=True))
            cv2.arrowedLine(frame, (W // 2, H // 2), (x, y), (255, 0, 255), 2)

            # расчет упавляющего воздействия через ПД регулятор
            # для удержания маркера в центре изображения
            err = pioutils.vec_length(error_vec)
            errsum += err
            u = k1 * err + k2 * (err - errold) + k3 * errsum
            u = round(min(u, 100))
            errold = err

            print(err, u)

            pioneer.vector_speed_control((u, error_vec_dir), (0, 0), min_val=0, max_val=100,
                                         use_polar=True, use_zy_xr_vectors=True, degrees=True)

            if err < 25:
                pioneer.send_rc_channels(1500, 1500, 1500, 1500, 1500, 2000)
                pioneer.step_inc()
        else:
            pioneer.vector_speed_control((0, 0), (0, 0), min_val=0, max_val=100,
                                         use_polar=True, use_zy_xr_vectors=True, degrees=True)

    if pioneer.step_get() == 3:
        pioneer.go_to_local_point(x=0, y=mvxy * col_cur, z=mvz * rows, callback=pioneer.step_inc)

    if pioneer.step_get() == 4:
        pioneer.sleep(1, callback=(lambda: add_marker(col_cur + (row_cur-1)*rows, markers_data), pioneer.step_inc))

    if pioneer.step_get() == 5:
        if col_cur == cols:
            col_cur = 0
            if row_cur != rows:
                row_cur += 1
                pioneer.step_reset(1)
            else:
                pioneer.step_inc()
        else:
            col_cur += 1
            pioneer.step_reset(3)

    if pioneer.step_get() == 6:
        pioneer.go_to_local_point(x=0, y=0, z=1, callback=pioneer.step_inc)

    if pioneer.step_get() == 7:
        pioneer.land()
        if pioneer.landed():
            break

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

database = np.array(database)
database.shape = (rows, cols)
print(database)
