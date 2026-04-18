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
        self._load_database_to_recognizer()

    def _load_database_to_recognizer(self):
        for face in self.database.get_all_faces():
            face_roi = np.array(face['encoding'], dtype=np.uint8)
            self.recognizer.train_images.append(face_roi)
            self.recognizer.train_labels.append(len(self.recognizer.known_face_names))
            self.recognizer.known_face_names.append(face['name'])
            self.recognizer.known_face_encodings.append(face_roi)

    def reload_faces(self):
        self.recognizer.clear_known_faces()
        for face in self.database.get_all_faces():
            image_path = face.get('image_path')
            if image_path and os.path.exists(image_path):
                self.recognizer.load_known_face(image_path, face['name'])
        if len(self.recognizer.train_images) > 0:
            self.recognizer.train()

    def register_face(self, image_path: str, name: str) -> bool:
        if self.recognizer.load_known_face(image_path, name):
            self.recognizer.train()
            encoding = self.recognizer.known_face_encodings[-1]
            self.database.add_face(name, encoding, image_path)
            return True
        return False

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

    def recognize_from_camera(self, camera_id: int = 0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Cannot open camera {camera_id}")
            return

        print("Press 'q' to quit, 's' to capture and recognize")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow('Smart Glasses Demo', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.recognizer.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    name, confidence, _ = self.recognizer.recognize_face(face_roi)
                    if name:
                        label = f"{name} (conf:{confidence:.0f})"
                        color = (0, 255, 0)
                    else:
                        label = f"Unknown (conf:{confidence:.0f})"
                        color = (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imshow('Recognition Result', frame)
                cv2.waitKey(0)
                cv2.destroyWindow('Recognition Result')

        cap.release()
        cv2.destroyAllWindows()

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
                print(f"  Face {i+1}: {result['name']} - Confidence: {result['confidence']:.1f}")

        elif command == 'camera':
            demo.recognize_from_camera()

        elif command == 'load-dir' and len(sys.argv) >= 3:
            dir_path = sys.argv[2]
            if not os.path.isabs(dir_path):
                dir_path = os.path.join(project_root, dir_path)
            print(f"Loading faces from: {dir_path}")
            for filename in os.listdir(dir_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    name = os.path.splitext(filename)[0]
                    image_path = os.path.join(dir_path, filename)
                    demo.register_face(image_path, name)

        elif command == 'list':
            demo.list_registered()

        elif command in ('help', '--help', '-h'):
            print("""
Smart Glasses Face Recognition Demo

Usage:
  python main.py register <image_path> <name>   - Register a new face
  python main.py recognize <image_path>         - Recognize faces in an image
  python main.py camera                         - Recognize from camera (live)
  python main.py load-dir <directory>           - Load all images from a directory
  python main.py list                           - List all registered faces
  python main.py help                           - Show this help message

Examples:
  python main.py register sample_faces/known/person1.jpg "Person 1"
  python main.py recognize sample_faces/unknown/test.jpg
  python main.py load-dir sample_faces/known
            """)

        else:
            print(f"Unknown command: {command}")
            print("Use 'python main.py help' for usage information.")

    else:
        print("""
===========================================
Smart Glasses Face Recognition Demo
===========================================

Commands:
  register <image_path> <name>   - Register a new face
  recognize <image_path>         - Recognize faces in an image
  camera                         - Recognize from camera (live)
  load-dir <directory>           - Load all images from a directory
  list                           - List all registered faces
  help                           - Show help message

Quick Start:
  1. Put known face images in demo/sample_faces/known/
  2. Run: python main.py load-dir demo/sample_faces/known
  3. Run: python main.py recognize demo/sample_faces/unknown/test.jpg
        """)


if __name__ == '__main__':
    main()
