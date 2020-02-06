from helpers import FACIAL_LANDMARKS_IDXS
from helpers import shape_to_np
import numpy as np
import cv2
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
# import argparse
import imutils
import dlib
import cv2

# detector = dlib.get_frontal_face_detector()
# p="shape_predictor_68_face_landmarks.dat"
# predictor = dlib.shape_predictor(p)
# fa = FaceAligner(predictor, desiredFaceWidth=256)

# cap = cv2.VideoCapture(0)
# while True:
#     _, image = cap.read()
#     image = imutils.resize(image, width=800)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # show the original input image and detect faces in the grayscale
#     # image
#     cv2.imshow("Input", image)
#     rects = detector(gray, 2)
#     for rect in rects:
#         # extract the ROI of the *original* face, then align the face
#         # using facial landmarks
#         (x, y, w, h) = rect_to_bb(rect)
#         faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
#         faceAligned = fa.align(image, gray, rect)
#         # angle=fa.align
#         # display the output images
#         cv2.imshow("Original", faceOrig)
#         cv2.imshow("Aligned", faceAligned)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#         cap.release()
#         cv2.destroyAllWindows()

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.3, 0.4),
                 desiredFaceWidth= 512, desiredFaceHeight=700):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        # print("inside align-->>")
        # exit()
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS[("left_eye")]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX))
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        # print(w,h)
        # exit()
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output
