import cv2
import os

student_id = input("Enter student ID: ").strip()
student_name = input("Enter student name: ").strip()

folder_name = f"{student_id}_{student_name}".replace(" ", "_")
save_path = os.path.join("dataset", folder_name)
os.makedirs(save_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
if face_cascade.empty():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
count = 0

print("Camera started. Look at the camera.")
print("Press ESC to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        file_name = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Images: {count}/30", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if count >= 30:
            break

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) == 27 or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Saved {count} images in {save_path}")
