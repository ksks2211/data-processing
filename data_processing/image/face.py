from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from PIL import Image

class MTFaceDetector:
    def __init__(self) -> None:
        self.detector = MTCNN()

    def detect(self, image_rgb):
        image = np.array(image_rgb, dtype=np.uint8)
        faces = self.detector.detect_faces(image)

        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return Image.fromarray(image)

class CV2FaceDetector:
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, image_rgb) -> bytes:
        image = np.array(image_rgb, dtype=np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return Image.fromarray(image_bgr)
