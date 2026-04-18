import os
import sys
import cv2
import numpy as np
from face_recognizer import FaceRecognizer
from face_db import FaceDatabase


class SmartGlassesDemo:
    def __init__(self, db_path=None):
        self.recognizer = FaceRecognizer()
        self.database = FaceDatabase(db_path)

    def register_face(self, image_path: str, name: str) -> bool:
        if self.recognizer.load_known_face(image_path, name):
            encoding = self.recognizer.known_face_encodings[-1]
            self.database.add_face(name, encoding, image_path)
            return True
        return False

    def reload_faces(self):
        self.recognizer.clear_known_faces()
        for face in self.database.get_all_faces():
            image_path = face.get('image_path')
            if image_path and os.path.exists(image_path):
                self.recognizer.load_known_face(image_path, face['name'])
        if len(self.recognizer.train_images) > 0:
            self.recognizer.train()

    def recognize_from_image(self, image_path: str) -> list:
        results = self.recognizer.recognize_all_faces(image_path)
        recognized = []
        for name, confidence, location in results:
            accuracy = self._lbph_confidence_to_accuracy(confidence)
            recognized.append({
                'name': name if name else 'Unknown',
                'confidence': confidence,
                'accuracy': accuracy,
                'location': location
            })
        return recognized

    def _lbph_confidence_to_accuracy(self, confidence: float) -> float:
        if confidence == 0:
            return 100.0
        if confidence > 150:
            return 0.0
        accuracy = max(0, (1 - confidence / 150) * 100)
        return accuracy

    def list_registered(self):
        names = self.database.get_names()
        if names:
            print("Registered faces:")
            for name in names:
                print(f"  - {name}")
        else:
            print("No faces registered yet.")
        return names


def main():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(demo_dir)

    demo = SmartGlassesDemo()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == 'register' and len(sys.argv) >= 4:
            image_path = sys.argv[2]
            name = sys.argv[3]
            if not os.path.isabs(image_path):
                image_path = os.path.join(project_root, image_path)
            demo.register_face(image_path, name)
        elif command == 'recognize' and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            if not os.path.isabs(image_path):
                image_path = os.path.join(project_root, image_path)
            results = demo.recognize_from_image(image_path)
            print("\nRecognition Results:")
            for i, result in enumerate(results):
                print(f"  Face {i+1}: {result['name']} - Accuracy: {result['accuracy']:.1f}%")
        elif command == 'list':
            demo.list_registered()
        elif command == 'help':
            print("""
Smart Glasses Face Recognition Demo

Usage:
  python main.py register <image_path> <name>
  python main.py recognize <image_path>
  python main.py list
  python main.py help
""")
    else:
        print("Smart Glasses Face Recognition Demo - use 'python main.py help'")


if __name__ == '__main__':
    main()
