"""
python3 video_to_images.py -i input_folder_path -o output_folder_path
"""

# Import required python libraries
import argparse
from multiprocessing import Process, Queue
import cv2
import queue
import os
import dlib
from glob import glob


# main Function that takes video as input and save frames into folder.
def test(tasks):
    for task in tasks:
        cap = cv2.VideoCapture(input_folder+"/" + str(task[0]) + ".avi")
        i = 1
        try:
            while True:
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray)
                # if there is single face in frame than it saves into face_detected folder
                if rects and len(rects) == 1:
                    cv2.imwrite(output_folder+"/face_detected/" + task[0] + "/" + task[0] + str(i) + ".jpg", frame)
                # if there is multiple face in frame than it saves into multi_face folder
                elif rects and len(rects) > 1:
                    cv2.imwrite(output_folder+"/multi_face/" + task[0] + "/" + task[0] + str(i) + ".jpg", frame)
                # if no face detected than it saves into No-face folder
                else:
                    cv2.imwrite(output_folder+"/No_face/" + task[0] + "/" + task[0] + str(i) + ".jpg", frame)
                i = i + 1
                cv2.imshow('output', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            cap.release()
        except Exception as e:
            # print(e)
            pass

# Multi Processing code that divide task between process.
def do_job(tasks_to_accomplish):
    while True:
        try:
            task = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            break
        else:
            test(task)
    return True



if __name__ == "__main__":
    # command line argument setup
    # text = "Description"
    # parser = argparse.ArgumentParser(description=text)
    # parser.add_argument("--input", "-i", help="Input Video file's path", type=str, required=True)
    # parser.add_argument("--output", "-o", help="Output Image file's path", type=str, required=True)

    # args = parser.parse_args()
    input_folder = "../input/video"
    output_folder = "../output/data"
    detector = dlib.get_frontal_face_detector()

    try:os.mkdir(output_folder+ '/No_face')
    except Exception as e:pass
    try:os.mkdir(output_folder+ '/face_detected')
    except Exception as e:pass
    try:os.mkdir(output_folder+ '/multi_face')
    except Exception as e:pass

    tasks_to_accomplish = Queue(maxsize=0)
    tasks_that_are_done = Queue(maxsize=0)
    temp=[]
    for folder in glob(input_folder+"/*"):
        file_name = folder.split("/")[-1].split(".")[0]
        try:os.mkdir(output_folder+"/face_detected/" + file_name)
        except Exception as e:pass
        try:os.mkdir(output_folder+"/No_face/" + file_name)
        except Exception as e:pass
        try:os.mkdir(output_folder+"/multi_face/" + file_name)
        except Exception as e:pass
        # print(folder,file_name)
        # exit()
        temp.append([file_name,folder])
        tasks_to_accomplish.put(temp)

    # define 5 process at time for fast result
    processes = [Process(target=do_job, args=(tasks_to_accomplish,)) for x in range(5)]

    # start all process
    for p in processes:
        p.start()

    # at the end join all process to main process
    for p in processes:
        p.join()

    print("All Video Files Completed")