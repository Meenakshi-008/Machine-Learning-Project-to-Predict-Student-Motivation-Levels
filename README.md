# Machine-Learning-Project-to-Predict-Student-Motivation-Levels

# 🎓 Student Motivation Level Prediction using Machine Learning

## 📖 Overview

Student motivation plays a critical role in academic success. This project uses Machine Learning techniques to predict a student's motivation level based on academic, personal, and extracurricular factors.

The system is developed using Python and Flask, providing a user-friendly web interface where users can enter student-related information and receive an instant prediction of motivation level.

The model classifies students into:

- 🟢 High Motivation
- 🟡 Medium Motivation
- 🔴 Low Motivation


## 🎯 Objectives

- Analyze factors influencing student motivation.
- Build a Machine Learning model for prediction.
- Develop a web-based application using Flask.
- Provide analytical insights into student behavior and performance.
- Demonstrate the practical use of Machine Learning in education.


## 🚀 Features

### Prediction System
✔ Predicts student motivation level

✔ Provides probability scores for each class

✔ Uses a trained Machine Learning model

✔ Real-time predictions through Flask

### Analytics Module
✔ Motivation distribution analysis

✔ Study-time statistics

✔ Feature importance analysis

✔ Correlation matrix generation

✔ Student demographic insights


## 📊 Input Parameters

The prediction is based on:

- Age
- Study Time
- Absences
- Tutoring Support
- Parental Support
- Extracurricular Activities
- Sports Participation
- Music Participation
- Volunteering Activities


## ⚙️ Feature Engineering

To improve prediction quality, additional features are generated:

### Study Efficiency

```text
Study Efficiency = Study Time / (Absences + 1)
```

### Support Score

```text
Support Score = Parental Support + Tutoring
```

### Activity Score

```text
Activity Score = Sports + Music + Volunteering
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Flask | Web Framework |
| NumPy | Numerical Computation |
| Joblib | Model Serialization |
| HTML | Frontend Development |
| CSS | User Interface Styling |
| Machine Learning | Prediction Model |
| Git | Version Control |
| GitHub | Project Hosting |


## 📁 Project Structure

```text
Machine-Learning-Project-to-Predict-Student-Motivation-Levels

│
├── app.py
├── model.py
├── model.pkl
├── scaler.pkl
├── columns.pkl
├── README.md
│
├── templates
│   └── index.html
│
├── requirements.txt
└── Procfile
```


## ▶️ Installation and Usage

### Clone Repository

```bash
git clone <repository-url>
```

### Navigate to Project Folder

```bash
cd Machine-Learning-Project-to-Predict-Student-Motivation-Levels
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

### Open Browser

```text
http://127.0.0.1:5000
```

## 📈 Model Output

The model predicts one of the following motivation levels:

| Prediction | Description |
|------------|-------------|
| High | Student shows strong motivation and engagement |
| Medium | Student demonstrates moderate motivation |
| Low | Student may require additional support and encouragement |


## 📸 Screenshots

### Home Page
(Add screenshot here)

### Prediction Result
(Add screenshot here)

### Analytics Dashboard
(Add screenshot here)


## 🔮 Future Enhancements

- Interactive dashboards using charts
- Improved prediction accuracy
- Real-world dataset integration
- Cloud deployment
- User authentication system
- Advanced Machine Learning algorithms


## 👨‍💻 Author

**Meenakshi Yendrapalli**

B.Tech Student | Machine Learning Enthusiast


## 🌟 Conclusion

This project demonstrates how Machine Learning and Flask can be combined to analyze student-related factors and predict motivation levels. It highlights the practical application of data science in the education sector and provides meaningful insights for understanding student engagement and performance.
