from PIL import Image
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import dlib
import cv2
from imutils import face_utils
import math
class Face:
    def __init__(self, image, gender='female'):
        self.image_pil = Image.open(image)
        self.image_plt = plt.imread(image)
        self.gender = gender
    def detectionMTCNN(self, image):
        dectector = MTCNN()
        return dectector.detect_faces(image)

    def resize(self, base_height=900):
        hpercent = (base_height / float(self.image_pil.size[1]))
        wsize = int((float(self.image_pil.size[0]) * float(hpercent)))
        return self.image_pil.resize((wsize, base_height), Image.ANTIALIAS)

    def rotation(self, image):
        coords = self.detectionMTCNN(image)
        left_eye = coords[0]['keypoints']['left_eye']
        right_eye = coords[0]['keypoints']['right_eye']
        dy = left_eye[1] - right_eye[1]
        dx = left_eye[0] - right_eye[0]
        angle = np.degrees(np.arctan2(dy, dx)) - 180
        return angle
    def crop(self, image):
        image_plt = plt.imread(image)
        image_pil = Image.open(image)
        coords = self.detectionMTCNN(image_plt)
        x, y, width, height = coords[0]['box']
        return image_pil.crop((x - 50, y - 50, x + width + 30, y + height + 50))
    def computeMask(self, image):
        if self.gender is 'female':
            maskpath= 'images/MaskFemale.png'
        elif self.gender is 'male':
            maskpath = 'images/MaskMale.png'
        image_plt = plt.imread(image)
        coords = self.detectionMTCNN(image_plt)
        right_eye = coords[0]['keypoints']['right_eye']
        right_mouth = coords[0]['keypoints']['mouth_right']
        mid_line = abs(right_eye[1]-right_mouth[1])
        baseheight = int(mid_line * 2.6)
        mask = Image.open(maskpath)
        hpercent = (baseheight / float(mask.size[1]))
        wsize = int((float(mask.size[0]) * float(hpercent)))
        #mask.resize((wsize, baseheight), Image.ANTIALIAS).save('Msk.png')
        return mask.resize((wsize, baseheight), Image.ANTIALIAS)
    def pasteMask(self, mask, face):
        coords = self.detectionMTCNN(face)
        left_eye = coords[0]['keypoints']['left_eye']
        right_eye = coords[0]['keypoints']['right_eye']
        right_mouth = coords[0]['keypoints']['mouth_right']

        dy = abs(left_eye[1] - right_eye[1])
        dx = abs(left_eye[0] - right_eye[0])
        desp_x = int(abs((left_eye[0] + dx/2)-(mask.size[0]/2)))
        desp_y = int(abs((right_mouth[1])-(mask.size[1]*0.7876)))
        face_computed_pil = Image.fromarray(face, 'RGB')
        face_computed_pil.paste(mask,(desp_x,desp_y),mask)
        return face_computed_pil
    def distances(self,image):
        p = "shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(p)
        face_computed_cv2 = cv2.imread(image)
        gray = cv2.cvtColor(face_computed_cv2, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        points = []
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            points.append(shape)
        points = points[0]
        return [math.sqrt((math.pow(points[0][0] - points[16][0],2))+ (math.pow(points[0][1] - points[16][1],2))), math.sqrt((math.pow(points[4][0] - points[12][0],2))+ (math.pow(points[4][1] - points[12][1],2))), math.sqrt((math.pow(points[27][0] - points[51][0],2))+ (math.pow(points[27][1] - points[51][1],2))), math.sqrt((math.pow(points[27][0] - points[8][0],2))+ (math.pow(points[27][1] - points[8][1],2)))]

    def main(self):
        self.image_pil = self.resize()
        self.image_pil =  self.image_pil.rotate(self.rotation(self.image_plt))
        self.image_pil.save('images/temp.jpg')
        self.image_pil = self.crop('images/temp.jpg')
        self.image_pil.save('images/temp.jpg')
        mask_computed = self.computeMask('images/temp.jpg')
        face_computed = plt.imread('images/temp.jpg')
        self.pasteMask(mask_computed, face_computed).save('finalImage.jpg', quality=100)
        return self.distances('images/temp.jpg')
