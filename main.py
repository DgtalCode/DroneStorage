import random

import cv2
import numpy as np
from pioneer_sdk import Pioneer
from pioneer_sdk import pioutils
import pygame

###################### ОБЪЯВЛЕНИЕ ПЕРЕМЕННЫХ ######################
# выбор источника видео:
# 0 - камера компьютера
# 1 - камера квадрокоптера
flag_video_source = 1

flag_rc_control = False

if flag_video_source == 1:
    pioneer = Pioneer(bad_connection_exit=False)
    if pioneer.bad_connection_occured():
        flag_video_source = 0

if flag_video_source == 0:
    cap = cv2.VideoCapture(0)

# параметры детектируемых маркеров
arDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)  # тип маркера
arPars = cv2.aruco.DetectorParameters_create()  # параметры детектирования (стандартные)

# позиционировнаие коптера
posx, posy, posz, yaw = 0, 0, 1.5, 0  # координаты коптера (м)
mvxy = 0.5  # шаг перемещений по Х У (м)
mvz = 0.2  # шаг перемещений по Z (м)
mvyaw = np.radians(10)  # шаг поворота (град)

# параметры ПД регулятора для выравниания по маркерам
k1 = 0.2  # реакция на отклонение
k2 = 0.1  # смягчение резких движений
err = 0  # ошибка (величина отклонения)
errold = 0  # старая ошибка (величина отклонения на предыдущей итерации)
u = 0  # управляющее воздействие

# ширина и высота изображения
W, H = (640, 480)
# флаг необходимости изменять размер изображения под установленный выше
flag_resize = False

control_w = 602
control_h = 300
left_stick_pos = ((control_w - 2) // 4, control_h // 2)
right_stick_pos = ((control_w - 2) // 4, control_h // 2)

pygame.init()
screen = pygame.display.set_mode((control_w, control_h))
pygame.display.set_caption('CONTROLS')
clock = pygame.time.Clock()


###################################################################


######################## ОБЪЯВЛЕНИЕ ФУНКЦИЙ #######################
def get_aruco_center(corners):
    """
    Считает координаты центра маркера по переданнм четырем углам
    :param corners: массив с коодинатами углов маркера
    :return: массив с координатами центра маркера
    """
    x = np.sum([c[0] for c in corners[0]]) // 4
    y = np.sum([c[1] for c in corners[0]]) // 4
    return int(x), int(y)


def get_frame():
    """
    Возвращает opencv фрейм от квадрокоптера или камеры ПК, в зависимоти от используемого источника
    :return: фрейм
    """
    global flag_video_source
    global frame
    if flag_video_source == 0:
        _, frame = cap.read()
    else:
        frame = cv2.imdecode(np.frombuffer(pioneer.get_raw_video_frame(), dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame


def nothing(x):
    pass

###################################################################


############################### MAIN ##############################
frame = get_frame()
if len(frame) != H:
    flag_resize = True

while True:
    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 0, 0), (control_w // 2, 0), (control_w // 2, control_h), 2)

    for event in pygame.event.get():
        if event.type == pygame.MOUSEMOTION:
            if event.buttons == (1, 0, 0):
                if event.pos[0] < control_w // 2:
                    left_stick_pos = event.pos
                    right_stick_pos = ((control_w - 2) // 4, control_h // 2)
                else:
                    right_stick_pos = (event.pos[0] - (control_w - 2) // 2, event.pos[1])
                    left_stick_pos = ((control_w - 2) // 4, control_h // 2)
            else:
                left_stick_pos = ((control_w - 2) // 4, control_h // 2)
                right_stick_pos = ((control_w - 2) // 4, control_h // 2)
        if event.type == pygame.WINDOWLEAVE:
            left_stick_pos = ((control_w - 2) // 4, control_h // 2)
            right_stick_pos = ((control_w - 2) // 4, control_h // 2)

    pygame.draw.circle(screen, (100, 100, 100), left_stick_pos, 20, 20)
    pygame.draw.circle(screen, (100, 100, 100), (right_stick_pos[0] + (control_w - 2) // 2, right_stick_pos[1]), 20, 20)
    pygame.display.flip()

    # считывание изображения
    frame = get_frame()
    if flag_resize:
        frame = cv2.resize(frame, (640, 480))

    # детектирование маркеров
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arDict, parameters=arPars)
    # отображение маркеров на изображении
    if corners:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids, (255, 0, 0))

    u, err, error_vec_dir = 0, 0, 0
    if len(corners) == 1:
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
        u = k1 * err - k2 * (err - errold)
        u = round(min(u, 100))
        errold = err

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('k'):
        print("k")
        break

    if flag_video_source == 1:
        if key == 32:
            print('space pressed')
            pioneer.arm()
        if key == 27:  # esc
            print('esc pressed')
            pioneer.disarm()
        if key == ord('g'):
            print('g pressed')
            flag_rc_control = not flag_rc_control
            print(f'Flag now is {flag_rc_control}')
        if key == ord('r'):
            pioneer.go_to_local_point(x=round(random.uniform(-1.5, 1.5), 2),
                                      y=round(random.uniform((-1.5, 1.5)), 2),
                                      z=round(random.uniform(1.0, 1.5), 2))

        # проверка режима полета: по скоростям или по точкам
        if flag_rc_control:
            # если пользователь не перехватывает управление, то позиционируемся по маркерам
            # в противном случае слушаемся пользователя
            if sum(left_stick_pos) + sum(right_stick_pos) != 600:
                pioneer.vector_speed_control(left_stick_pos, right_stick_pos, min_val=0, max_val=300,
                                             rev_left_x=True, rev_right_x=True)
            else:
                pioneer.vector_speed_control((u, error_vec_dir), (0, 0), min_val=0, max_val=100,
                                             use_polar=True, use_zy_xr_vectors=True, degrees=True)
        else:
            # важнейший параметр - 6й канал в 2000, так как возвращает коптер в режим полета по точкам
            pioneer.send_rc_channels(channel_1=1500, channel_2=1500, channel_3=1500,
                                     channel_4=1500, channel_5=1500, channel_6=2000)

cv2.destroyAllWindows()
cap.release()
