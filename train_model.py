import cv2
import os
import numpy as np

data_path = "dataset"

faces = []
labels = []
label_map = {}
current_label = 0

for person in os.listdir(data_path):
    person_path = os.path.join(data_path, person)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

labels = np.array(labels)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

model.save("face_model.xml")

# Save label map
np.save("labels.npy", label_map)

print("Training completed!")
