import json
import os
from datetime import datetime
from typing import List, Optional, Dict
import numpy as np


class FaceDatabase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "data", "face_database.json")
        self.db_path = db_path
        self.faces = []
        self.load_db()

    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.faces = data.get('faces', [])
            except Exception as e:
                print(f"Error loading database: {e}")
                self.faces = []
        else:
            self.faces = []

    def save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump({'faces': self.faces, 'updated': datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False

    def add_face(self, name: str, encoding: np.ndarray, image_path: str = None) -> bool:
        flat_encoding = encoding.flatten().tolist()
        for face in self.faces:
            if face['name'] == name:
                print(f"Face with name '{name}' already exists. Updating...")
                face['encoding'] = flat_encoding
                if image_path:
                    face['image_path'] = image_path
                face['updated_time'] = datetime.now().isoformat()
                return self.save_db()

        self.faces.append({
            'name': name,
            'encoding': flat_encoding,
            'image_path': image_path,
            'added_time': datetime.now().isoformat(),
            'updated_time': datetime.now().isoformat()
        })
        return self.save_db()

    def remove_face(self, name: str) -> bool:
        initial_len = len(self.faces)
        self.faces = [f for f in self.faces if f['name'] != name]
        if len(self.faces) < initial_len:
            return self.save_db()
        return False

    def get_all_faces(self) -> List[Dict]:
        return self.faces

    def get_encoding_by_name(self, name: str) -> Optional[np.ndarray]:
        for face in self.faces:
            if face['name'] == name:
                return np.array(face['encoding'])
        return None

    def get_names(self) -> List[str]:
        return [f['name'] for f in self.faces]

    def clear_all(self) -> bool:
        self.faces = []
        return self.save_db()

    def get_count(self) -> int:
        return len(self.faces)
