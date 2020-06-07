import cv2
import numpy as np
import dlib
from math import hypot


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/Gebruiker/PycharmProjects/EyeGaze/shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_lenght / ver_line_lenght
    return ratio


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]

    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


frames = 0
letter_index = 0

while True:

    _, frame = cap.read()
    #new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        # Detect blinking

        #  first row
        cv2.rectangle(frame, (10,    25),  (100, 100),   (0, 255, 0), 5)# first rectangle
        cv2.rectangle(frame, (560,  25),  (650, 100),    (0, 255, 0), 5)# second rectangle
        cv2.rectangle(frame, (1110, 25),  (1200, 100),   (0, 255, 0), 5)# third rectangle

        #  mid row
        cv2.rectangle(frame, (10,   325),  (100,  400),   (255, 255, 0), 5)# first rectangle
        cv2.rectangle(frame, (560,  325),  (650,  400),   (255, 255, 0), 5)# second rectangle
        cv2.rectangle(frame, (1110, 325),  (1200, 400),   (255, 255, 0), 5)# third rectangle

        #  last row
        cv2.rectangle(frame, (10,   625),  (100,  700),   (0, 255, 255), 5)# first rectangle
        cv2.rectangle(frame, (560,  625),  (650,  700),   (0, 255, 255), 5)# second rectangle
        cv2.rectangle(frame, (1110, 625),  (1200, 700),   (0, 255, 255), 5)# third rectangle



        # rectagle third row
        #cv2.rectangle(frame, (0,   400),   (100, 100), (0, 255, 0), 5)
        #cv2.rectangle(frame, (250, 400), (350, 100), (0, 255, 0), 5)
        #cv2.rectangle(frame, (500, 400), (600, 100), (0, 255, 0), 5)
        # left eye
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        # right eye
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        # blink of eyes
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        # landmarks
        landmarks = predictor(gray, face)
        # ----------------- temple ---------------------
        temple = (landmarks.part(0).x, landmarks.part(0).y)
        cv2.circle(frame, temple, 4, (255, 0, 0), -5)
        # -------------    eye conrer -----------------
        eye_corner = (landmarks.part(37).x, landmarks.part(37).y)
        cv2.circle(frame, eye_corner, 4, (255, 0, 0), -5)

       # if blinking_ratio > 5.7:
          #  cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
        # Gaze detection

        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        vertical = ""

        print(int(temple[1])- int(eye_corner[1]))
        if temple[1] - eye_corner[1] <= -10:
            #cv2.putText(frame, "looking down", (10, 300), font, 2, (0, 0, 255), 3)
            if gaze_ratio <=2:
                cv2.putText(frame, "RIGHT", (1000, 500), font, 2, (0, 0, 255), 3)
                cv2.putText(frame, str(int(gaze_ratio)), (1000, 550), font, 2, (0, 0, 255), 3)
                cv2.rectangle(frame, (1110, 625), (1200, 700), (0, 255, 255), -1)  # third rectangle
                #  new_frame[:] = (0, 0, 255)
            elif gaze_ratio > 6 :
                #  new_frame[:] = (255, 0, 0)
                cv2.putText(frame, "LEFT", (50, 500), font, 2, (0, 0, 255),  3)
                cv2.putText(frame, str(int(gaze_ratio)), (50, 550), font, 2, (0, 0, 255),  3)
                cv2.rectangle(frame, (10, 625), (100, 700), (0, 255, 255), -1)  # first rectangle
            else :
                cv2.putText(frame, "CENTER",  (500, 500), font, 2, (0, 0, 255), 3)
                cv2.putText(frame, str(int(gaze_ratio)),  (500, 550), font, 2, (0, 0, 255), 3)
                cv2.rectangle(frame, (560, 625), (650, 700), (0, 255, 255), -1)  # second rectangle

        elif (temple[1] - eye_corner[1]) >= 20:
            if gaze_ratio <= 1:
                cv2.putText(frame, "RIGHT", (1000, 30), font, 2, (0, 0, 255), 3)
                cv2.putText(frame, str(int(gaze_ratio)), (1000, 50), font, 2, (0, 0, 255), 3)
                cv2.rectangle(frame, (1110, 25), (1200, 100), (0, 255, 0), -1)  # third rectangle

            elif 1 < gaze_ratio < 1.7:
                cv2.putText(frame, "CENTER", (400, 30), font, 2, (0, 0, 255), 3)
                cv2.putText(frame, str(int(gaze_ratio)), (400, 50), font, 2, (0, 0, 255), 3)
                cv2.rectangle(frame, (560, 25), (650, 100), (0, 255, 0), -1)  # second rectangle
            else:
                cv2.putText(frame, "LEFT", (200, 30), font, 2, (0, 0, 255), 3)
                cv2.putText(frame, str(int(gaze_ratio)), (200, 50), font, 2, (0, 0, 255), 3)
                cv2.rectangle(frame, (10, 25), (100, 100), (0, 255, 0), -1)  # first rectangle

        else:
            # cv2.putText(frame, "looking up", (10, 300), font, 2, (0, 0, 255), 3)
            if gaze_ratio <= 1:
                cv2.putText(frame, "RIGHT", (1000, 400), font, 2, (0, 0, 255), 3)
                cv2.putText(frame, str(int(gaze_ratio)), (1000, 450), font, 2, (0, 0, 255), 3)
                cv2.rectangle(frame, (1110, 325), (1200, 400), (255, 255, 0), -1)  # third rectangle

            elif 1 < gaze_ratio < 1.7:
                cv2.putText(frame, "CENTER", (400, 400), font, 2, (0, 0, 255), 3)
                cv2.putText(frame, str(int(gaze_ratio)), (400, 450), font, 2, (0, 0, 255), 3)
                cv2.rectangle(frame, (560, 325), (650, 400), (255, 255, 0), -1)  # second rectangle
            else:
                cv2.putText(frame, "LEFT", (100, 400), font, 2, (0, 0, 255), 3)
                cv2.putText(frame, str(int(gaze_ratio)), (100, 450), font, 2, (0, 0, 255), 3)
                cv2.rectangle(frame, (10, 325), (100, 400), (255, 255, 0), -1)  # first rectangle






    if frames == 10:
        letter_index = +1
        frame = 0
    for i in range(15):
        if i == 5:
            light = True
        else:
            light = False

    # drawProductsBoad()
    cv2.imshow("Frame", frame)
  #  cv2.imshow("New frame", new_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
#  cv2.imshow("products", products)


cap.release()
cv2.destroyAllWindows()
