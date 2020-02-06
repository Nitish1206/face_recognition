"""
python3 master.py -m 215_labelled_25-12-18 -c web -t False

"""
# Import Required python libraries
import cv2
import face_recognition
import numpy as np
from keras.engine.saving import load_model
import argparse
import json
import dlib
# from load_trt_model_for_recognition import get_prediction
import traceback
import os
import sys
from helpers import *
from face_alignment import FaceAligner
# Command Line Argument Setup
text = 'Description'
parser = argparse.ArgumentParser(description = text)
# parser.add_argument("--model", "-m", help="name of your model", type=str,required=True)
parser.add_argument("--camera", "-c", help="ip or web", type=str, default="web")
parser.add_argument("--terminal", "-t", help="Print result in terminal True or False", default=False,)
parser.add_argument("--threshold", "-th", help="set the Threshold",type=float,default=0.98,)
args = parser.parse_args()

def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    python3 = sys.executable
    os.execl(python3, python3, * sys.argv)
# Function For get result from model,
def result_from_model(face_data,model,label):
    face_encode = np.asarray([face_data])
    result = model.predict(face_encode)
    index=np.argmax(result[0])
    # label_model = model.predict_classes(face_encode)
    accuracy = result[0][index]
    person = label[str(index)]
    return person,accuracy


# Main Function
def main(model,cam_in,label):
    # load the face detection model
    detector=dlib.get_frontal_face_detector()
    face_cascade = cv2.CascadeClassifier('./input/haarcascade_frontalface_default.xml')
    # Takes camera input 0 for default
    cap = cv2.VideoCapture(cam_in)
    frame_width, frame_height = 320, 240
    cap.set(3, frame_width)
    cap.set(4, frame_height)
    detector = dlib.get_frontal_face_detector()
    p = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(p)
    faz = FaceAligner(predictor)
    # cap.set(35, -68.0)
    # Set Resolution of camera
    # cap.set(3, 720)
    # cap.set(4, 1280)
    # Main Process Start from here.
    while True:
        # Read frame by frame from camera live stream
        ret, image = cap.read()
        # Convert BGR image to gray scale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects=detector(gray,0)
        if len(rects)>0:
            for rect in rects:
                img=faz.align(image, gray, rect)
        # Convert BGR image to RGB for Face Encodings
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # faces = face_cascade.detectMultiScale(gray,1.3, 5)   # Function returns number of faces in frame
        # boxes = face_recognition.face_locations(rgb, model='cnn')
                faces_predicted=[]
        # for face in faces:
                face_encodings = face_recognition.face_encodings(rgb)
        # face_encodings = face_recognition.face_encodings(rgb[face[1]:face[3]+face[1],face[0]:face[2]+face[0]])  # Function Returns Face Encodings
            if(len(face_encodings))>0:
                person, accuracy = result_from_model(face_encodings[0], model, label)
                faces_predicted.append((person,accuracy,rect_to_bb(rect)))

        for single_face in faces_predicted:
            x,y,w,h=single_face[2]
            if accuracy>THRESHOLD:    # if accuracy is greater than threshold value than name will be printed on frame
                cv2.rectangle(image, (x, y), (x + w, y + h),(255,0,0), 2)
                cv2.putText(image, str(single_face[0])+" "+str(single_face[1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0,255,0), 2)
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
        cv2.imshow("output", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):

            cv2.destroyAllWindows()
            cap.release()
            answer = input("Do you want to restart this program ? ")
            if answer.lower().strip() in "y yes".split():
                restart_program()
            else:
                break



if __name__ == '__main__':
    try:
        # command line argument setup
        # if args.model:
        model_name = "aligned_test_320_240_512_700"
        camera = args.camera
        THRESHOLD = args.threshold
        terminal = args.terminal
        model = load_model("src/model/"+model_name+".model")
        if camera == "ip":
            cam_in = "rtsp://admin:abc12345@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
        elif camera == "web":
            cam_in = 0
        # path="/home/nitish/Nitish/work/face_recognition_ankie_flow/src/model/2020-01-27851720.model"
        with open("src/model_label/"+model_name+".json", 'r') as fp:
            label = json.load(fp)
        main(model,cam_in,label)
    except Exception as e:
        print("Error-->",traceback.print_exc())
