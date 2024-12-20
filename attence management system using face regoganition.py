import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime

# Database Setup
def setup_database():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        time TEXT
    )
    """)
    conn.commit()
    conn.close()

def mark_attendance(name):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
    conn.commit()
    conn.close()

# Face Detection and Recognition
def load_trained_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    return recognizer

def detect_and_recognize():
    recognizer = load_trained_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Define the names associated with trained IDs
    names = ["Unknown", "Mohanraj G"]

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 50:  # Confidence threshold for recognition
                name = names[face_id]
                mark_attendance(name)
                color = (0, 255, 0)  # Green for recognized faces
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unrecognized faces

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Recognition - Attendance", frame)

        if cv2.waitKey(1) == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Train the Model
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    path = "dataset"
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        face_id = int(os.path.split(image_path)[-1].split(".")[1])  # Format: User.ID.extension
        faces = face_cascade.detectMultiScale(gray_image)

        for (x, y, w, h) in faces:
            face_samples.append(gray_image[y:y+h, x:x+w])
            ids.append(face_id)

    recognizer.train(face_samples, np.array(ids))
    recognizer.write("trainer.yml")
    print("Model training complete!")

# Main Program
if __name__ == "__main__":
    setup_database()
    print("Choose an option:")
    print("1. Train Model")
    print("2. Recognize Faces")
    choice = input("Enter your choice: ")

    if choice == "1":
        train_model()
    elif choice == "2":
        detect_and_recognize()
    else:
        print("Invalid choice!")
