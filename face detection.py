import cv2
import dlib
import numpy

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/Gebruiker/PycharmProjects/EyeGaze/shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # landmarks
        landmarks = predictor(gray, face)
        #temple
        temple  = (landmarks.part(0).x,landmarks.part(0).y)
        cv2.circle(frame, temple, 4, (255, 0, 0), -5)
        #eye conrer
        eye_corner = (landmarks.part(37).x,landmarks.part(37).y)
        cv2.circle(frame, eye_corner, 4, (255, 0, 0), -5)


        if temple[1] < eye_corner[1]:
            cv2.putText(frame, "looking down", (10, 300), font, 2,(0, 0, 255), 3)
        elif temple[1] > eye_corner[1]:
            cv2.putText(frame, "looking up", (10, 300), font, 2,(0, 0, 255), 3)
        else:
            cv2.putText(frame, "looking center", (10, 300), font, 2,(0, 0, 255), 3)


        #for n in range(34, 42):
        #    x = landmarks.part(n).x
        #    y = landmarks.part(n).y
          #  cv2.circle(frame, (x, y), 4, (255, 0, 0), -5)
          #  print(y1, y2)
       # print(landmarks.part(33).x)
       # cv2.putText(frame,  str(y1-y), (200, 100), font, 2, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
