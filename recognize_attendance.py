import cv2
import os
import csv
import numpy as np
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountkey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()


def save_to_firebase(student_id, name):
    db.collection("attendance").add({
        "student_id": student_id,
        "name": name,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.xml")

label_map = np.load("labels.npy", allow_pickle=True).item()

attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["StudentID", "Name", "Date", "Time"])

marked_today = set()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = model.predict(face)

        if confidence < 70:
            person = label_map[label]
            parts = person.split("_", 1)
            student_id = parts[0]
            name = parts[1] if len(parts) > 1 else "Unknown"

            text = f"{student_id} - {name}"

            today = datetime.now().strftime("%Y-%m-%d")
            now_time = datetime.now().strftime("%H:%M:%S")
            unique_key = f"{student_id}_{today}"

            if unique_key not in marked_today:
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([student_id, name, today, now_time])

                save_to_firebase(student_id, name)

                marked_today.add(unique_key)
                print(f"Attendance marked for {student_id} {name}")

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
