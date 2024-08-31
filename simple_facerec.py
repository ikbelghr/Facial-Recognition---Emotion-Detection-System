import sqlite3
import cv2
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.preprocessing.image import img_to_array
import face_recognition
import glob
import os
from emotion_model import load_emotion_model  # Import the function to load the emotion model

# Define emotion labels
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.frame_resizing = 0.25
        self.image_limit = 1  # Number of images to save per appearance
        self.appearance_time_window = timedelta(minutes=10)  # Time window for appearance
        self.emotion_model = load_emotion_model('model_file.h5')  # Load emotion detection model

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            if filename not in self.known_face_ids:
                self.known_face_ids.append(filename)
                self.known_face_encodings.append([img_encoding])
            else:
                index = self.known_face_ids.index(filename)
                self.known_face_encodings[index].append(img_encoding)

        print("Encoding images loaded")

    def save_known_faces_to_db(self):
        conn = sqlite3.connect('customer_visits.db')
        c = conn.cursor()
        for person_id, encodings in zip(self.known_face_ids, self.known_face_encodings):
            for encoding in encodings:
                c.execute(
                    'INSERT OR REPLACE INTO known_faces (customer_id, face_encoding) VALUES (?, ?)',
                    (person_id, encoding.tobytes())
                )
        conn.commit()
        conn.close()

    def load_known_faces_from_db(self):
        conn = sqlite3.connect('customer_visits.db')
        c = conn.cursor()
        c.execute('SELECT customer_id, face_encoding FROM known_faces')
        rows = c.fetchall()
        for row in rows:
            person_id = row[0]
            encoding = np.frombuffer(row[1], dtype=np.float64)
            if person_id not in self.known_face_ids:
                self.known_face_ids.append(person_id)
                self.known_face_encodings.append([encoding])
            else:
                index = self.known_face_ids.index(person_id)
                self.known_face_encodings[index].append(encoding)
        conn.close()

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_ids = []
        emotions = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = []
            face_distances = []
            for encodings in self.known_face_encodings:
                match = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)
                distance = face_recognition.face_distance(encodings, face_encoding)
                matches.append(match)
                face_distances.append(distance)

            name = "Unknown"
            if any([True in match for match in matches]):
                best_match_index = np.argmin([min(distance) for distance in face_distances])
                name = self.known_face_ids[best_match_index]
            else:
                new_id = f"ID_{len(self.known_face_ids) + 1}"
                self.known_face_ids.append(new_id)
                self.known_face_encodings.append([face_encoding])
                self.save_known_faces_to_db()
                name = new_id

            face_ids.append(name)

            # Predict emotion
            y1, x2, y2, x1 = face_location
            face_image = small_frame[y1:y2, x1:x2]
            face_image_resized = cv2.resize(face_image, (48, 48))
            face_image_gray = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            face_image_array = img_to_array(face_image_gray)
            face_image_array = np.expand_dims(face_image_array, axis=0) / 255.0
            emotion_prediction = self.emotion_model.predict(face_image_array)
            emotion_label_index = np.argmax(emotion_prediction)
            emotion_label = emotion_labels.get(emotion_label_index, "Unknown")
            emotions.append(emotion_label)

            # Save face image with timestamp and emotion
            now = datetime.now()
            last_appearance = self.get_last_appearance(name)

            if last_appearance is None or now - last_appearance > self.appearance_time_window:
                if self.get_image_count(name, now) < self.image_limit:
                    y1, x2, y2, x1 = face_location
                    face_image = small_frame[y1:y2, x1:x2]
                    self.save_face_image(name, face_image, now, emotion_label)
                    # Update the dominant emotion
                    self.update_dominant_emotion(name, emotions)

        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_ids, emotions

    def update_dominant_emotion(self, customer_id, emotions):
        if not emotions:
            return
        
        # Determine the most frequent emotion
        emotion_count = {label: emotions.count(label) for label in set(emotions)}
        dominant_emotion = max(emotion_count, key=emotion_count.get)

        # Update the database with the dominant emotion
        conn = sqlite3.connect('customer_visits.db')
        c = conn.cursor()
        now = datetime.now()
        c.execute(
            'INSERT INTO visit_emotions (customer_id, timestamp, dominant_emotion) VALUES (?, ?, ?)',
            (customer_id, now.isoformat(), dominant_emotion)
        )
        conn.commit()
        conn.close()

    def save_face_image(self, customer_id, face_image, timestamp, emotion_label):
        _, img_encoded = cv2.imencode('.jpg', face_image)
        img_blob = img_encoded.tobytes()

        conn = sqlite3.connect('customer_visits.db')
        c = conn.cursor()
        c.execute(
            'INSERT INTO face_images (customer_id, image, timestamp, emotion) VALUES (?, ?, ?, ?)',
            (customer_id, img_blob, timestamp.isoformat(), emotion_label)
        )
        conn.commit()
        conn.close()

    def get_image_count(self, customer_id, current_time):
        conn = sqlite3.connect('customer_visits.db')
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM face_images WHERE customer_id = ? AND timestamp >= ?', 
                  (customer_id, (current_time - self.appearance_time_window).isoformat()))
        count = c.fetchone()[0]
        conn.close()
        return count

    def get_last_appearance(self, customer_id):
        conn = sqlite3.connect('customer_visits.db')
        c = conn.cursor()
        c.execute('SELECT MAX(timestamp) FROM face_images WHERE customer_id = ?', (customer_id,))
        last_appearance = c.fetchone()[0]
        conn.close()
        if last_appearance:
            return datetime.fromisoformat(last_appearance)
        else:
            return None
