import cv2
from datetime import datetime
import dlib
import os
from detect_blur import detect_blur
from train import call_train
import time
from face_alignment import FaceAligner

name=input("enter name")

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

try: os.mkdir("../output/data"+"/"+name)
except Exception as e: pass

frame_width,frame_height=320,240
detector = dlib.get_frontal_face_detector()
p="shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)
faz=FaceAligner(predictor)

frame_counter=0
cap=cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)
# face_cascade = cv2.CascadeClassifier('./input/haarcascade_frontalface_default.xml')
# out = cv2.VideoWriter("../input/video/"+name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
name_counter=0
while True:
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_text=detect_blur(gray)
    if blur_text=="Not Blurry":
        rects = detector(gray,0)
        if len(rects)==1:
            img = faz.align(frame, gray, rects[0])

            cv2.imwrite("../output/data"+"/"+name+"/"+str(name_counter)+".jpg", img)
            name_counter+=1

    if name_counter==160:
        break
    cv2.imshow("real",frame)
    cv2.waitKey(1)
cap.release()
# out.release()
cv2.destroyAllWindows()

time.sleep(10)
call_train("../output/data"+"/"+name,name)