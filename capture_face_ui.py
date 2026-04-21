import cv2
import os

student_id = os.environ.get("STUDENT_ID", "").strip()
student_name = os.environ.get("STUDENT_NAME", "").strip()

if not student_id or not student_name:
    print("Missing student details")
    raise SystemExit

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

path = f"dataset/{student_id}_{student_name}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

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

        count += 1
        file_name = f"{path}/{count}.jpg"
        cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{count}/30", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if count >= 30:
            break

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) == 27 or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()

print("Images saved!")
