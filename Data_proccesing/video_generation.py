import cv2
from datetime import datetime
import dlib
import os
from .detect_blur import detect_blur
from ..src.train import call_train
import time
name=input("enter name")

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

try: os.mkdir("../output/data"+"/"+name)
except Exception as e: pass

detector = dlib.get_frontal_face_detector()
frame_counter=0
cap=cv2.VideoCapture(0)
frame_width,frame_height=640,480
# out = cv2.VideoWriter("../input/video/"+name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
name_counter=0
while True:
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_text=detect_blur(gray)
    if blur_text=="Not Blurry":
        rects = detector(gray,0)
        if rects and len(rects) == 1:
            cv2.imwrite("../output/data"+"/"+name+"/"+str(name_counter)+".jpg", frame)
            name_counter+=1
        # for (i, rect) in enumerate(rects):
        #     (x, y, w, h) = rect_to_bb(rect)

        # out.write(frame)

        frame_counter=frame_counter+1
        # print(frame_counter)
        if frame_counter==120:
            break
    cv2.imshow("real",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
# out.release()
cv2.destroyAllWindows()

time.sleep(10)
call_train("../output/data"+"/"+name)