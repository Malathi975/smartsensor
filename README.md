# 🌱 Smart Farming Sensor Analysis – Detect Agricultural Operating Conditions

A beginner-friendly Machine Learning mini-project that analyzes smart farming sensor values and detects agricultural operating/crop health conditions using a **Random Forest Classifier** and an interactive **Streamlit** web application.

---

## 📌 Project Overview
- **Project Title:** Smart Farming Sensor Analysis – Detect Agricultural Operating Conditions
- **Goal:** Provide farmers and agricultural operators with an easy-to-use, AI-powered system that interprets real-time sensor measurements (soil moisture, pH, temperature, rainfall, humidity, sunlight, NDVI, pesticide usage) and instantly classifies the field's agricultural operating condition (`Healthy`, `Mild`, `Moderate`, `Severe`).
- **Core Technologies:** Python, Pandas, NumPy, Scikit-Learn, Streamlit, Matplotlib, Seaborn.

---

## 📊 Dataset Description
- **Dataset File:** `Smart_Farming_Crop_Yield_2024.csv`
- **Total Records (Rows):** 500
- **Total Attributes (Columns):** 22
- **Selected Numerical Sensor Features (8 Features):**
  1. `soil_moisture_%` – Soil moisture content percentage (10.16% – 44.98%)
  2. `soil_pH` – Soil pH level (5.51 – 7.50)
  3. `temperature_C` – Ambient farm temperature in Celsius (15.02°C – 39.97°C)
  4. `rainfall_mm` – Precipitation level in millimeters (100.18 mm – 1198.81 mm)
  5. `humidity_%` – Relative air humidity percentage (30.01% – 89.96%)
  6. `sunlight_hours` – Daily sunlight duration (4.01 – 11.99 hours)
  7. `NDVI_index` – Normalized Difference Vegetation Index (0.30 – 0.90)
  8. `pesticide_usage_ml` – Chemical/pesticide volume applied (50.12 – 999.85 ml)
- **Target Output Column:** `crop_disease_status`
  - Classes: `Healthy`, `Mild`, `Moderate`, `Severe`
  - Missing values (130 samples without disease symptoms) are handled and labeled as `Healthy`.

---

## 🤖 Machine Learning Model
- **Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Dataset Split:** 80% Training (400 samples), 20% Testing (100 samples) with `random_state=42` to ensure reproducible results and prevent data leakage.
- **Evaluation Metrics:** Test Accuracy, Confusion Matrix, and Classification Report.

---

## 📂 Project Directory Structure

```text
miniproject/
│
├── app.py                            # Streamlit web application
├── Smart_Farming_Crop_Yield_2024.csv  # Kaggle smart farming sensor dataset
├── requirements.txt                  # List of Python dependencies
└── README.md                         # Project documentation and report
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Make sure Python 3.10+ is installed on your system.

### 2. Install Required Packages
Open your terminal/command prompt in the project directory and run:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Application

Launch the Streamlit web dashboard with the following command:

```bash
streamlit run app.py
```

The interactive application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🖥️ Application Features & Expected Output

1. **📊 Dataset & Model Overview:** Real-time metrics showing total records (500), number of sensor features (8), target column (`crop_disease_status`), and model accuracy.
2. **🌾 Interactive Sensor Inputs:** Adjust sliders in the sidebar for soil moisture, pH, temperature, rainfall, humidity, sunlight, NDVI index, and pesticide usage.
3. **🔮 Real-Time Condition Detection:** Click **"Detect Agricultural Condition"** to obtain the predicted agricultural status (`Healthy`, `Mild`, `Moderate`, `Severe`), actionable recommendations, and class probability breakdown.
4. **📈 Visualizations & Diagnostics:**
   - Agricultural condition class distribution
   - Confusion matrix heatmap
   - Sensor feature distribution plots
   - Random Forest feature importance chart
5. **📋 Dataset Explorer:** Searchable dataset table and statistical summary.

---

# 🎓 College Mini-Project Submission Section

### 🎯 AIM
> "To develop a machine learning-based Smart Farming Sensor Analysis System that detects agricultural operating conditions using farming sensor and environmental data."

### 📝 DEFINE (Problem Statement)
> "How might we develop a simple machine learning application that uses farming sensor data to identify agricultural operating conditions and provide an understandable result to the user?"

### ⚠️ EXISTING PROBLEMS
1. **Sensor Complexity:** Raw numerical sensor readings (e.g. NDVI index, moisture percentages) are difficult for farmers to interpret manually.
2. **Time-Consuming Manual Monitoring:** Manual physical inspection of large fields is labor-intensive and delays critical interventions.
3. **Unpredictable Environmental Variations:** Fluctuations in humidity, rainfall, and temperature can rapidly degrade crop health without immediate detection.
4. **Absence of Simple Decision-Support Tools:** Lack of accessible, lightweight software for small-to-medium farming operations.

### 🎯 OBJECTIVES
1. Collect and inspect farming sensor data from the Kaggle dataset.
2. Preprocess sensor data (handle missing values, remove duplicates, select key features).
3. Train a Random Forest classification model to learn relationships between sensor readings and operating conditions.
4. Detect the agricultural operating condition accurately in real time.
5. Display the result clearly through an intuitive, green-themed Streamlit user interface.

### 🤝 EMPATHY (Understanding User Needs)
Farmers and agricultural field supervisors need straightforward, actionable insights rather than raw decimal numbers or complex math. When sensor readings indicate high moisture combined with high temperature and dropping NDVI, they need to know immediately whether the crop condition is **Healthy**, **Mild**, **Moderate**, or **Severe**, along with clear recommendations on what steps to take next.

### 💡 IDEATE (Proposed Solution)
The proposed solution builds a supervised machine learning pipeline using Python and Scikit-Learn. A **Random Forest Classifier** is selected because it excels at capturing non-linear relationships across diverse environmental sensors (soil, climate, vegetation index) without requiring complex neural networks. The model is wrapped in a **Streamlit** web app that offers real-time slider controls and instant condition detection.

### 🛠️ PROTOTYPE (Streamlit Dashboard Layout)
The prototype interface is structured into three main components:
1. **Sidebar Input Controls:** Intuitive sliders calibrated to the exact dataset minimum and maximum sensor ranges.
2. **Main Detection View:** Key dataset metrics, "Detect Agricultural Condition" trigger button, color-coded result cards, and prediction probability charts.
3. **Analytics & Explorer Tabs:** Interactive charts (distribution plots, confusion matrix, feature importance) and raw dataset tables.

### ⚡ ALGORITHM (5 Steps)
1. **Read the Smart Farming Dataset:** Load `Smart_Farming_Crop_Yield_2024.csv` using pandas.
2. **Select and Prepare Sensor Features:** Clean missing values, encode target classes, and select 8 numerical sensor columns.
3. **Train Machine Learning Classification Model:** Split data (80/20) and fit a Random Forest Classifier.
4. **Enter Sensor Values and Predict Condition:** Receive user input from Streamlit sliders and feed the vector to the trained model.
5. **Display Predicted Condition:** Render color-coded condition badges, confidence percentages, and visual charts.

### 📊 RESULT
The Smart Farming Sensor Analysis application was successfully developed and tested on the 500-record dataset. The Random Forest Classifier was trained using 8 numerical sensor features (`soil_moisture_%`, `soil_pH`, `temperature_C`, `rainfall_mm`, `humidity_%`, `sunlight_hours`, `NDVI_index`, `pesticide_usage_ml`) to predict the 4 agricultural operating condition classes (`Healthy`, `Mild`, `Moderate`, `Severe`). The Streamlit web dashboard allows seamless real-time sensor testing, interactive visualization, and clear diagnostic outputs.
