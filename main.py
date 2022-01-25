import cv2
import numpy as np
from pioneer_sdk import Pioneer
import pygame

###################### ОБЪЯВЛЕНИЕ ПЕРЕМЕННЫХ ######################
# выбор источника видео:
# 0 - камера компьютера
# 1 - камера квадрокоптера
flag_video_source = 1

flag_new_command = False

if flag_video_source == 1:
    pioneer = Pioneer(bad_connection_exit=False)
    if pioneer.bad_connection_occured():
        flag_video_source = 0

if flag_video_source == 0:
    cap = cv2.VideoCapture(0)

# параметры детектируемых маркеров
arDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)  # тип маркера
arPars = cv2.aruco.DetectorParameters_create()  # параметры детектирования (стандартные)

# позиционировнаие коптера
posx, posy, posz, yaw = 0, 0, 0, 0  # координаты коптера (м)
mvxy = 0.2  # шаг перемещений по Х У (м)
mvz = 0.1  # шаг перемещений по Z (м)
mvyaw = np.radians(10)  # шаг поворота (град)

# параметры ПД регулятора для выравниания по маркерам
k1 = 0.0008  # реакция на отклонение
k2 = 0.002  # смягчение резких движений
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
    '''
    Calculates marker center coordinates from its corners
    :param corners: array with 4 corners
    :return: array with integer marker center coordinates (x,y)
    '''
    x = np.sum([c[0] for c in corners[0]]) // 4
    y = np.sum([c[1] for c in corners[0]]) // 4
    return int(x), int(y)


def vec_from_points(point_start, point_end):
    '''
    Creates a vector from two points
    :param point_start: array with start point coordinates (x,y)
    :param point_end: array with end point coordinates (x,y)
    :return: vector as an array (x,y)
    '''
    return (point_end[0] - point_start[0],
            point_start[1] - point_end[1])


def vec_length(vec):
    '''
    Returns length of a vector
    :param vec: vector as an array (x,y)
    :return: float: vector length
    '''
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def vec_direction(vec, to_degrees=False):
    '''
    Returns an angle between vector and X-axis
    :param vec: vector as an array (x,y)
    :param to_degrees: flag that turns the output to degrees instead of radians
    :return: float: angle between vector and X-axis
    '''
    angle = np.arctan2(vec[1], vec[0])
    if to_degrees:
        angle = np.degrees(angle)
    return angle


def distance_between_points(point1, point2):
    '''
    Returns a distance between two points
    :param point1: array with coordinates (x,y)
    :param point2: array with coordinates (x,y)
    :return: float: distance between two points
    '''
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def get_frame():
    global flag_video_source
    global frame
    if flag_video_source == 0:
        _, frame = cap.read()
    else:
        frame = cv2.imdecode(np.frombuffer(pioneer.get_raw_video_frame(), dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame


def nothing(x):
    pass


def remap(value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((value - old_min) * new_range) / old_range) + new_min
    return new_value


def copter_vector_speed_control(left_vector=[0, 0], right_vector=[0, 0], min_val=-500, max_val=500,
                                polar=False, degrees=True, rev_left_y=False, rev_left_x=False,
                                rev_right_y=False, rev_right_x=False):
    def __s(val):
        return 1 if val else -1

    if min_val != -500 or max_val != 500 and not polar:
        right_vector = tuple(map(lambda x: remap(x, min_val, max_val, -500, 500), right_vector))
        left_vector = tuple(map(lambda x: remap(x, min_val, max_val, -500, 500), left_vector))
    if polar:
        if max_val != 500:
            right_vector = (remap(right_vector[0], 0, max_val, 0, 500), right_vector[1])
            left_vector = (remap(left_vector[0], 0, max_val, 0, 500), left_vector[1])
        if degrees:
            right_vector = (right_vector[0], np.radians(right_vector[1]))
            left_vector = (left_vector[0], np.radians(left_vector[1]))

        right_vector = (right_vector[0] * np.cos(-right_vector[1]),
                        right_vector[0] * np.sin(-right_vector[1]))
        left_vector = (left_vector[0] * np.cos(-left_vector[1]),
                       left_vector[0] * np.sin(-left_vector[1]))

    channel_1 = round(1500 + left_vector[1] * __s(rev_left_y))
    channel_2 = round(1500 - left_vector[0] * __s(rev_left_x))
    channel_3 = round(1500 - right_vector[1] * __s(rev_right_y))
    channel_4 = round(1500 + right_vector[0] * __s(rev_right_x))

    print(channel_1, channel_2, channel_3, channel_4)

    pioneer.send_rc_channels(channel_1=channel_1, channel_2=channel_2, channel_3=channel_3,
                             channel_4=channel_4, channel_5=2000, channel_6=1000)


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

    if len(corners) == 1:
        # расчет центра маркера
        x, y = get_aruco_center(corners[0])
        cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS)

        # расчет отклонения маркера от центра изображения в виде вектора
        error_vec = vec_from_points((W // 2, H // 2), (x, y))
        error_vec_dir = vec_direction(error_vec)
        cv2.arrowedLine(frame, (W // 2, H // 2), (x, y), (255, 0, 255), 2)

        # расчет упавляющего воздействия через ПД регулятор
        # для удержания маркера в центре изображения
        err = vec_length(error_vec)
        u = k1 * err - k2 * (err - errold)
        errold = err

        # расчет коректировчных смещений коптера
        x_correction = round(u * np.cos(error_vec_dir), 2)
        y_correction = round(u * np.sin(error_vec_dir), 2)
        print(x_correction, y_correction)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('t'):
        break

    if flag_video_source == 1:
        if key == 32:
            print('space pressed')
            pioneer.arm()
            print('point')
        if key == 27:  # esc
            print('esc pressed')
            pioneer.disarm()

        copter_vector_speed_control(left_stick_pos, right_stick_pos, min_val=0, max_val=300,
                                    rev_left_x=True, rev_right_x=True)
        # copter_vector_speed_control((0, 0), (500, 180), polar=True)

cv2.destroyAllWindows()
cap.release()
