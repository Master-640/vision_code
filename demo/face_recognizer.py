import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_cascade = None
        self.recognizer = None
        self._init_opencv()

    def _init_opencv(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8,
            threshold=100
        )
        self.train_labels = []
        self.train_images = []
        self.is_trained = False

    def load_known_face(self, image_path: str, name: str) -> bool:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return False

        try:
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"Cannot read image: {image_path}")
                return False

            faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))

            if len(faces) == 0:
                print(f"No face detected in: {image_path}")
                return False

            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]

            self.train_images.append(face_roi)
            self.train_labels.append(len(self.known_face_names))

            self.known_face_names.append(name)
            self.known_face_encodings.append(face_roi)
            return True
        except Exception as e:
            print(f"Error loading face from {image_path}: {e}")
            return False

    def train(self):
        if len(self.train_images) > 0:
            self.recognizer.train(self.train_images, np.array(self.train_labels))
            self.is_trained = True

    def recognize_face(self, gray_face: np.ndarray) -> Tuple[Optional[str], float, int]:
        if not self.is_trained or len(self.known_face_names) == 0:
            return None, float('inf'), -1

        label, confidence = self.recognizer.predict(gray_face)
        name = self.known_face_names[label]
        return name, float(confidence), int(label)

    def detect_faces(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []

        image = cv2.imread(image_path)
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))

        return [tuple(f) for f in faces]

    def recognize_all_faces(self, image_path: str) -> List[Tuple[str, float, Tuple]]:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []

        if not self.is_trained:
            self.train()

        image = cv2.imread(image_path)
        if image is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))

        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            name, confidence, _ = self.recognize_face(face_roi)
            results.append((name, confidence, (y, x+w, y+h, x)))

        return results

    def get_face_count(self) -> int:
        return len(self.known_face_names)

    def clear_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.train_images = []
        self.train_labels = []
        self.is_trained = False
