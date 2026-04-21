from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os
import shutil
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountkey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()


@app.route("/")
def home():
    docs = db.collection("attendance").stream()

    rows = []
    unique_students = set()

    for doc in docs:
        data = doc.to_dict()
        student_id = data.get("student_id", "")
        name = data.get("name", "")
        time = data.get("time", "")

        rows.append({
            "student_id": student_id,
            "name": name,
            "time": time
        })

        if student_id:
            unique_students.add(student_id)

    rows = sorted(rows, key=lambda x: x["time"], reverse=True)

    total_count = len(rows)
    total_students = len(unique_students)
    latest_rows = rows[:5]

    return render_template(
        "index.html",
        total_count=total_count,
        total_students=total_students,
        latest_rows=latest_rows
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        student_name = request.form.get("student_name", "").strip()

        env = os.environ.copy()
        env["STUDENT_ID"] = student_id
        env["STUDENT_NAME"] = student_name

        # save student in Firebase
        try:
            db.collection("students").document(student_id).set({
                "student_id": student_id,
                "name": student_name
            })
        except Exception as e:
            print("Firebase student save error:", e)

        subprocess.run(["python3", "capture_face_ui.py"], cwd=BASE_DIR, env=env)

        return render_template("message.html", message="Face capture completed!")

    return render_template("register.html")


@app.route("/train")
def train():
    subprocess.run(["python3", "train_model.py"], cwd=BASE_DIR)
    return render_template("message.html", message="Training completed!")


@app.route("/recognize")
def recognize():
    subprocess.run(["python3", "recognize_attendance.py"], cwd=BASE_DIR)
    return render_template("message.html", message="Attendance session ended!")


@app.route("/attendance")
def attendance():
    docs = db.collection("attendance").stream()

    rows = []
    for doc in docs:
        data = doc.to_dict()
        rows.append({
            "student_id": data.get("student_id", ""),
            "name": data.get("name", ""),
            "time": data.get("time", "")
        })

    rows = sorted(rows, key=lambda x: x["time"], reverse=True)

    return render_template("attendance.html", rows=rows, total_count=len(rows))


@app.route("/students")
def students():
    student_list = []

    dataset_path = os.path.join(BASE_DIR, "dataset")
    if os.path.exists(dataset_path):
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):
                parts = folder.split("_", 1)
                student_id = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                student_list.append({
                    "student_id": student_id,
                    "name": name,
                    "folder": folder
                })

    student_list = sorted(student_list, key=lambda x: x["student_id"])
    return render_template("students.html", students=student_list)


@app.route("/delete_student/<folder_name>", methods=["POST"])
def delete_student(folder_name):
    folder_path = os.path.join(BASE_DIR, "dataset", folder_name)

    student_id = folder_name.split("_", 1)[0]

    # delete local face folder
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

    # delete Firebase student record
    try:
        db.collection("students").document(student_id).delete()
    except Exception as e:
        print("Firebase delete error:", e)

    # retrain model
    try:
        subprocess.run(["python3", "train_model.py"], cwd=BASE_DIR)
    except Exception as e:
        print("Retrain error:", e)

    return redirect(url_for("students"))


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
