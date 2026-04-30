import os
from flask import Flask, render_template, request, redirect, session
import pickle
import numpy as np
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)
app.secret_key = "secret123"

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            password TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            age INTEGER,
            glucose REAL,
            bp REAL,
            insulin REAL,
            bmi REAL,
            skin_thickness REAL,
            dpf REAL,
            result TEXT
        )
    ''')

    conn.commit()
    conn.close()


def save_prediction(username, age, glucose, bp, insulin, bmi, skin_thickness, dpf, result):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute(
        "INSERT INTO predictions (username, age, glucose, bp, insulin, bmi, skin_thickness, dpf, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (username, age, glucose, bp, insulin, bmi, skin_thickness, dpf, result)
    )

    conn.commit()
    conn.close()


# ================= MODEL LOAD =================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(base_dir, 'Pickle', 'model_new.pkl')
scaler_path = os.path.join(base_dir, 'Pickle', 'scaler_new.pkl')

scaler = pickle.load(open(scaler_path, 'rb'))
model = pickle.load(open(model_path, 'rb'))

# ================= DATASET (for graph) =================
df = pd.read_csv(os.path.join(base_dir, "diabetes.csv"))

# DB init
init_db()


# ================= GRAPH FUNCTIONS =================
def plot_feature_importance(model):
    try:
        importance = model.feature_importances_
        features = ["Glucose", "BP", "SkinThickness", "Insulin", "BMI", "DPF", "Age"]

        plt.figure()
        plt.barh(features, importance)
        plt.xlabel("Importance")
        plt.title("Feature Importance")

        plt.savefig("static/feature.png")
        plt.close()
    except:
        print("Model does not support feature importance")


def plot_bmi(df):
    plt.figure()
    plt.scatter(df["BMI"], df["Outcome"])
    plt.xlabel("BMI")
    plt.ylabel("Diabetes")

    plt.savefig("static/bmi.png")
    plt.close()


# ================= ROUTES =================

@app.route('/')
def home():
    return render_template("index.html")


# LOGIN
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            session["user"] = username
            return redirect('/')
        else:
            return "Invalid Login"

    return render_template("login.html")


# SIGNUP
@app.route('/signup', methods=['POST'])
def signup():
    username = request.form["username"]
    password = request.form["password"]

    hashed_password = generate_password_hash(password)

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

    return redirect('/login')


# RESULT
@app.route('/result', methods=['POST'])
def result():
    try:
        Age = int(request.form.get("Age"))
        Glucose = int(request.form.get("Glucose"))
        BloodPressure = int(request.form.get("BloodPressure"))
        Insulin = int(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        SkinThickness = int(request.form.get("SkinThickness"))
        DPF = float(request.form.get("DiabetesPedigreeFunction"))

        if (
            Age <= 0 or Glucose <= 0 or BloodPressure <= 0 or
            Insulin < 0 or BMI <= 0 or SkinThickness < 0 or DPF < 0
        ):
            return render_template("error.html", error="Invalid Input!")

        input_data = [
            Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DPF, Age
        ]

        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        result_text = "Diabetic" if prediction == 1 else "Not Diabetic"

        username = session.get("user", "guest")

        save_prediction(
            username,
            Age,
            Glucose,
            BloodPressure,
            Insulin,
            BMI,
            SkinThickness,
            DPF,
            result_text
        )

        return render_template('result.html', prediction=result_text)

    except Exception as e:
        return render_template('error.html', error=str(e))


# HISTORY
@app.route('/history')
def history():
    username = session.get("user", "guest")

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT * FROM predictions WHERE username=?", (username,))
    data = c.fetchall()

    conn.close()

    return render_template("history.html", data=data)


# DASHBOARD
@app.route('/dashboard')
def dashboard():
    plot_feature_importance(model)
    plot_bmi(df)

    return render_template("dashboard.html")


# LOGOUT
@app.route('/logout')
def logout():
    session.pop("user", None)
    return redirect('/')


# RUN
if __name__ == '__main__':
    app.run(debug=True)