"""
=============================================================================
🌱 SMART FARMING SENSOR ANALYSIS – DETECT AGRICULTURAL OPERATING CONDITIONS
=============================================================================
An enhanced, beginner-friendly Streamlit web application using machine learning
(Random Forest Classifier) on real smart farming sensor data to analyze
field conditions and detect agricultural operating/health conditions.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ENHANCED THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Farming Sensor Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enhanced CSS for modern agricultural glassmorphic UI
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
    /* Global Font and Body */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(46, 125, 50, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #e8f5e9;
        font-weight: 400;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #f1f8e9;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    /* Metric Cards */
    .stat-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-align: left;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(46, 125, 50, 0.08);
        border-color: #a5d6a7;
    }
    .stat-icon {
        font-size: 1.4rem;
        margin-bottom: 6px;
    }
    .stat-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1b5e20;
        line-height: 1.1;
    }
    .stat-lbl {
        font-size: 0.82rem;
        font-weight: 600;
        color: #616161;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Result Banner */
    .result-box {
        border-radius: 16px;
        padding: 24px;
        margin-top: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }
    .result-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* Custom Styled Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        color: #ffffff !important;
        font-size: 1.05rem;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.65rem 1.6rem;
        border: none;
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.25);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #388e3c 0%, #2e7d32 100%);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.35);
        transform: translateY(-1px);
    }

    /* Preset Buttons */
    .preset-chip {
        background: #f1f8e9;
        border: 1px solid #c5e1a5;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 0.85rem;
        color: #2e7d32;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Section Cards */
    .section-card {
        background: #ffffff;
        border: 1px solid #e8ede7;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
CSV_FILENAME = "Smart_Farming_Crop_Yield_2024.csv"

# Actual numerical sensor features from dataset
FEATURE_COLS = [
    'soil_moisture_%',
    'soil_pH',
    'temperature_C',
    'rainfall_mm',
    'humidity_%',
    'sunlight_hours',
    'NDVI_index',
    'pesticide_usage_ml'
]

# Friendly display names & units
FEATURE_INFO = {
    'soil_moisture_%': {'label': 'Soil Moisture', 'unit': '%', 'icon': '💧'},
    'soil_pH': {'label': 'Soil pH', 'unit': 'pH', 'icon': '🧪'},
    'temperature_C': {'label': 'Temperature', 'unit': '°C', 'icon': '🌡️'},
    'rainfall_mm': {'label': 'Rainfall', 'unit': 'mm', 'icon': '🌧️'},
    'humidity_%': {'label': 'Humidity', 'unit': '%', 'icon': '💨'},
    'sunlight_hours': {'label': 'Sunlight Duration', 'unit': 'hrs/day', 'icon': '☀️'},
    'NDVI_index': {'label': 'NDVI Vegetation Index', 'unit': 'Index (0-1)', 'icon': '🌿'},
    'pesticide_usage_ml': {'label': 'Pesticide Application', 'unit': 'ml', 'icon': '🛡️'}
}

TARGET_COL = 'crop_disease_status'

@st.cache_data
def load_and_preprocess_data():
    """Load dataset, handle missing values, and verify features."""
    df = pd.read_csv(CSV_FILENAME)
    df = df.drop_duplicates()
    
    # Handle missing values: null indicates healthy/unaffected condition
    df[TARGET_COL] = df[TARGET_COL].fillna('Healthy')
    
    for col in FEATURE_COLS:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
            
    return df

df = load_and_preprocess_data()


# -----------------------------------------------------------------------------
# 3. MODEL TRAINING & EVALUATION
# -----------------------------------------------------------------------------
@st.cache_resource
def train_model(data):
    """Trains Random Forest Classifier and computes performance metrics."""
    X = data[FEATURE_COLS]
    y = data[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return rf_model, accuracy, cm, report, rf_model.classes_, X_test, y_test

model, accuracy, cm, report, class_labels, X_test, y_test = train_model(df)


# -----------------------------------------------------------------------------
# 4. HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <h1 class="hero-title">🌱 Smart Farming Sensor Analysis</h1>
                <p class="hero-sub">AI-Powered Agricultural Operating Condition & Crop Health Detection</p>
            </div>
            <div class="hero-badge">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00e676; margin-right: 8px;"></span>
                Random Forest Model Active
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. METRIC STAT CARDS (SECTION A)
# -----------------------------------------------------------------------------
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">📑</div>
            <div class="stat-val">{len(df)}</div>
            <div class="stat-lbl">Dataset Records</div>
        </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">📡</div>
            <div class="stat-val">{len(FEATURE_COLS)}</div>
            <div class="stat-lbl">Sensor Features</div>
        </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-val" style="font-size: 1.5rem; word-break: break-all;">{TARGET_COL}</div>
            <div class="stat-lbl">Target Variable</div>
        </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">⚡</div>
            <div class="stat-val">{accuracy * 100:.1f}%</div>
            <div class="stat-lbl">Test Accuracy</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")


# -----------------------------------------------------------------------------
# 6. SIDEBAR: SENSOR INPUT CONTROLS & PRESET SCENARIOS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🌾 Preset Scenarios")
st.sidebar.caption("Click a preset to quickly load realistic field conditions:")

# Session state initialization for smooth preset switching
preset_defaults = {
    'soil_moisture': float(df['soil_moisture_%'].mean()),
    'soil_pH': float(df['soil_pH'].mean()),
    'temperature': float(df['temperature_C'].mean()),
    'rainfall': float(df['rainfall_mm'].mean()),
    'humidity': float(df['humidity_%'].mean()),
    'sunlight': float(df['sunlight_hours'].mean()),
    'ndvi': float(df['NDVI_index'].mean()),
    'pesticide': float(df['pesticide_usage_ml'].mean())
}

for key, val in preset_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Preset Buttons Grid
p_col1, p_col2 = st.sidebar.columns(2)
with p_col1:
    if st.button("🌿 Optimal Field", help="Load ideal growing conditions"):
        st.session_state['soil_moisture'] = 32.0
        st.session_state['soil_pH'] = 6.5
        st.session_state['temperature'] = 24.0
        st.session_state['rainfall'] = 600.0
        st.session_state['humidity'] = 65.0
        st.session_state['sunlight'] = 8.5
        st.session_state['ndvi'] = 0.82
        st.session_state['pesticide'] = 350.0
        st.rerun()

    if st.button("💧 High Moisture", help="Excess moisture/high humidity conditions"):
        st.session_state['soil_moisture'] = 42.5
        st.session_state['soil_pH'] = 6.2
        st.session_state['temperature'] = 29.0
        st.session_state['rainfall'] = 1100.0
        st.session_state['humidity'] = 86.0
        st.session_state['sunlight'] = 5.0
        st.session_state['ndvi'] = 0.52
        st.session_state['pesticide'] = 680.0
        st.rerun()

with p_col2:
    if st.button("☀️ Drought Stress", help="High temperature, low moisture"):
        st.session_state['soil_moisture'] = 12.0
        st.session_state['soil_pH'] = 7.3
        st.session_state['temperature'] = 38.5
        st.session_state['rainfall'] = 120.0
        st.session_state['humidity'] = 32.0
        st.session_state['sunlight'] = 11.5
        st.session_state['ndvi'] = 0.35
        st.session_state['pesticide'] = 200.0
        st.rerun()

    if st.button("⚠️ Pest Outbreak", help="Elevated pesticide & stress factors"):
        st.session_state['soil_moisture'] = 22.0
        st.session_state['soil_pH'] = 5.7
        st.session_state['temperature'] = 33.0
        st.session_state['rainfall'] = 450.0
        st.session_state['humidity'] = 78.0
        st.session_state['sunlight'] = 7.0
        st.session_state['ndvi'] = 0.40
        st.session_state['pesticide'] = 920.0
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Fine-Tune Sensor Inputs")

# Interactive Sliders tied to session state
soil_moisture = st.sidebar.slider(
    "💧 Soil Moisture (%)",
    min_value=float(df['soil_moisture_%'].min()),
    max_value=float(df['soil_moisture_%'].max()),
    value=float(st.session_state['soil_moisture']),
    step=0.1,
    help="Volumetric soil water content percentage."
)

soil_pH = st.sidebar.slider(
    "🧪 Soil pH",
    min_value=float(df['soil_pH'].min()),
    max_value=float(df['soil_pH'].max()),
    value=float(st.session_state['soil_pH']),
    step=0.01,
    help="Soil acidity/alkalinity measure (optimal: 6.0 - 7.0)."
)

temperature = st.sidebar.slider(
    "🌡️ Temperature (°C)",
    min_value=float(df['temperature_C'].min()),
    max_value=float(df['temperature_C'].max()),
    value=float(st.session_state['temperature']),
    step=0.1,
    help="Ambient farm temperature."
)

rainfall = st.sidebar.slider(
    "🌧️ Rainfall (mm)",
    min_value=float(df['rainfall_mm'].min()),
    max_value=float(df['rainfall_mm'].max()),
    value=float(st.session_state['rainfall']),
    step=1.0,
    help="Total season/period rainfall."
)

humidity = st.sidebar.slider(
    "💨 Humidity (%)",
    min_value=float(df['humidity_%'].min()),
    max_value=float(df['humidity_%'].max()),
    value=float(st.session_state['humidity']),
    step=0.1,
    help="Relative atmospheric humidity."
)

sunlight = st.sidebar.slider(
    "☀️ Sunlight Hours (hrs/day)",
    min_value=float(df['sunlight_hours'].min()),
    max_value=float(df['sunlight_hours'].max()),
    value=float(st.session_state['sunlight']),
    step=0.1,
    help="Daily hours of active sunlight."
)

ndvi = st.sidebar.slider(
    "🌿 NDVI Index (Vegetation Health)",
    min_value=float(df['NDVI_index'].min()),
    max_value=float(df['NDVI_index'].max()),
    value=float(st.session_state['ndvi']),
    step=0.01,
    help="Normalized Difference Vegetation Index (higher = healthier canopy)."
)

pesticide = st.sidebar.slider(
    "🛡️ Pesticide Usage (ml)",
    min_value=float(df['pesticide_usage_ml'].min()),
    max_value=float(df['pesticide_usage_ml'].max()),
    value=float(st.session_state['pesticide']),
    step=1.0,
    help="Volume of applied pest control chemicals."
)


# -----------------------------------------------------------------------------
# 7. MAIN TABS INTERFACE
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Condition Detection",
    "📈 Visualizations & Analytics",
    "📋 Dataset Explorer",
    "🎓 Project Architecture & Report"
])


# =============================================================================
# TAB 1: CONDITION PREDICTION & AGRONOMIC ADVISORY
# =============================================================================
with tab1:
    st.markdown("### 🌾 Live Sensor Input Summary")
    
    # Input Vector DataFrame
    input_df = pd.DataFrame([[
        soil_moisture,
        soil_pH,
        temperature,
        rainfall,
        humidity,
        sunlight,
        ndvi,
        pesticide
    ]], columns=FEATURE_COLS)

    # 4-Column Sensor Summary Cards
    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        st.metric("Soil Moisture", f"{soil_moisture:.1f} %", delta="Optimal: 20-35%")
        st.metric("Soil pH", f"{soil_pH:.2f}", delta="Optimal: 6.0-7.2")
    with ic2:
        st.metric("Temperature", f"{temperature:.1f} °C", delta="Optimal: 20-30°C")
        st.metric("Rainfall", f"{rainfall:.0f} mm")
    with ic3:
        st.metric("Humidity", f"{humidity:.1f} %", delta="Optimal: 50-70%")
        st.metric("Sunlight", f"{sunlight:.1f} hrs")
    with ic4:
        ndvi_health = "🟢 Dense" if ndvi > 0.65 else ("🟡 Moderate" if ndvi > 0.45 else "🔴 Sparse")
        st.metric("NDVI Index", f"{ndvi:.2f}", delta=ndvi_health)
        st.metric("Pesticide", f"{pesticide:.0f} ml")

    st.write("")
    
    # Section C: Prediction Button
    predict_btn = st.button("Detect Agricultural Condition", use_container_width=True)

    # If button clicked or already in session
    if predict_btn:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        confidence = max(prediction_proba) * 100

        # Detailed agronomic mapping
        condition_meta = {
            "Healthy": {
                "color": "#1b5e20",
                "bg": "#e8f5e9",
                "badge_bg": "#c8e6c9",
                "title": "🟢 Condition: Healthy / Optimal Operating Environment",
                "status_desc": "All sensor indicators are within safe agronomic boundaries. Crop physiology and soil moisture levels reflect normal vegetative growth.",
                "actions": [
                    "Continue routine drip/sprinkler irrigation schedule.",
                    "Maintain weekly soil moisture and NDVI drone/satellite monitoring.",
                    "No immediate chemical interventions required."
                ]
            },
            "Mild": {
                "color": "#e65100",
                "bg": "#fff3e0",
                "badge_bg": "#ffe0b2",
                "title": "🟡 Condition: Mild Stress / Early Disease Risk",
                "status_desc": "Subtle physiological stress or early pathogenous conditions detected. Favorable moisture/humidity balance may be slightly off-target.",
                "actions": [
                    "Inspect crop roots and leaf undersides for localized fungal spots.",
                    "Regulate irrigation frequency to avoid localized waterlogging.",
                    "Consider mild organic foliar spray or bio-fungicide treatment."
                ]
            },
            "Moderate": {
                "color": "#d84315",
                "bg": "#fbe9e7",
                "badge_bg": "#ffccbc",
                "title": "🟠 Condition: Moderate Stress / Active Disease Detected",
                "status_desc": "Substantial deviation in sensor readings (elevated temperature/humidity or reduced NDVI). Crop vigor is compromised.",
                "actions": [
                    "Apply targeted agronomic pest/disease mitigation immediately.",
                    "Calibrate soil pH and apply secondary micronutrients (Zinc/Boron).",
                    "Isolate heavily affected crop zones to stop contagion."
                ]
            },
            "Severe": {
                "color": "#b71c1c",
                "bg": "#ffebee",
                "badge_bg": "#ffcdd2",
                "title": "🔴 Condition: Severe Agricultural Stress / Critical Disease Alert",
                "status_desc": "Critical operating hazard detected! Severe vegetation degradation, extreme moisture imbalance, or high infection severity.",
                "actions": [
                    "Initiate emergency agricultural intervention protocol immediately.",
                    "Consult local agronomist and conduct in-field pathogen lab testing.",
                    "Apply corrective broad-spectrum crop protection measures."
                ]
            }
        }

        meta = condition_meta.get(prediction, condition_meta["Healthy"])

        # Section D: Result Display Card
        st.markdown(f"""
            <div class="result-box" style="background-color: {meta['bg']}; border-left: 8px solid {meta['color']};">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span class="result-badge" style="background-color: {meta['badge_bg']}; color: {meta['color']};">
                            {meta['title']}
                        </span>
                        <h2 style="color: {meta['color']}; margin: 6px 0 10px 0; font-size: 1.8rem; font-weight: 800;">
                            🌱 Detected Agricultural Condition: {prediction}
                        </h2>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.85rem; color: #555;">Model Confidence</span>
                        <div style="font-size: 1.8rem; font-weight: 800; color: {meta['color']};">{confidence:.1f}%</div>
                    </div>
                </div>
                <p style="font-size: 1rem; color: #333; margin-bottom: 14px; line-height: 1.5;">
                    <b>Diagnostic Assessment:</b> {meta['status_desc']}
                </p>
                <div style="background: rgba(255,255,255,0.7); padding: 14px 18px; border-radius: 10px; border: 1px solid rgba(0,0,0,0.05);">
                    <b style="color: {meta['color']}; font-size: 0.95rem;">📋 Recommended Field Actions:</b>
                    <ul style="margin-top: 6px; margin-bottom: 0; padding-left: 20px; font-size: 0.92rem; color: #424242;">
                        {''.join([f'<li style="margin-bottom: 4px;">{act}</li>' for act in meta['actions']])}
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Probability Distribution Bar Chart
        st.markdown("#### 🎯 Class Probability Breakdown")
        proba_df = pd.DataFrame({
            "Operating Condition": class_labels,
            "Probability (%)": [round(p * 100, 2) for p in prediction_proba]
        })
        
        # Render visual probability bars
        p_cols = st.columns(len(class_labels))
        for idx, row in proba_df.iterrows():
            c_label = row["Operating Condition"]
            c_val = row["Probability (%)"]
            with p_cols[idx]:
                st.metric(f"{c_label}", f"{c_val:.1f}%")
                st.progress(float(c_val / 100.0))

        st.write("")
        # Report Download Feature
        report_text = f"""SMART FARMING SENSOR ANALYSIS REPORT
===================================================
Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Application: Smart Farming Sensor Analysis

SENSOR READINGS:
---------------------------------------------------
- Soil Moisture      : {soil_moisture:.2f} %
- Soil pH            : {soil_pH:.2f}
- Temperature        : {temperature:.2f} °C
- Rainfall           : {rainfall:.2f} mm
- Humidity           : {humidity:.2f} %
- Sunlight Duration  : {sunlight:.2f} hrs/day
- NDVI Index         : {ndvi:.2f}
- Pesticide Usage    : {pesticide:.2f} ml

DIAGNOSTIC RESULT:
---------------------------------------------------
Detected Condition   : {prediction}
Prediction Confidence: {confidence:.2f} %

RECOMMENDED ACTIONS:
---------------------------------------------------
{chr(10).join([f"- {a}" for a in meta['actions']])}

===================================================
Developed with Python, Scikit-Learn, and Streamlit
"""
        st.download_button(
            label="📥 Download Diagnostic Report (.txt)",
            data=report_text,
            file_name=f"Smart_Farming_Report_{prediction}.txt",
            mime="text/plain"
        )


# =============================================================================
# TAB 2: VISUALIZATIONS & MODEL DIAGNOSTICS (SECTION 4)
# =============================================================================
with tab2:
    st.markdown("### 📈 Visual Analytics & Machine Learning Insights")
    st.caption("Gain deep visibility into sensor correlations, model classification performance, and feature significance.")

    v_row1_col1, v_row1_col2 = st.columns(2)

    with v_row1_col1:
        st.markdown("#### 1. Target Class Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(
            x=TARGET_COL,
            hue=TARGET_COL,
            data=df,
            palette=["#2e7d32", "#66bb6a", "#ffa726", "#ef5350"],
            legend=False,
            order=df[TARGET_COL].value_counts().index,
            ax=ax1
        )
        ax1.set_title("Agricultural Operating Condition Distribution", fontsize=11, fontweight='bold', pad=12)
        ax1.set_xlabel("Condition Class", fontsize=10)
        ax1.set_ylabel("Number of Farm Records", fontsize=10)
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        sns.despine(top=True, right=True)
        plt.tight_layout()
        st.pyplot(fig1)

    with v_row1_col2:
        st.markdown("#### 2. Confusion Matrix Heatmap (Test Set)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='YlGn',
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax2
        )
        ax2.set_title("Confusion Matrix (Actual vs Predicted)", fontsize=11, fontweight='bold', pad=12)
        ax2.set_xlabel("Predicted Condition", fontsize=10, fontweight='600')
        ax2.set_ylabel("Actual Condition", fontsize=10, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    v_row2_col1, v_row2_col2 = st.columns(2)

    with v_row2_col1:
        st.markdown("#### 3. Sensor Feature Importance")
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            "Sensor Feature": [FEATURE_INFO[f]['label'] for f in FEATURE_COLS],
            "RawFeature": FEATURE_COLS,
            "Importance": importances
        }).sort_values("Importance", ascending=True)

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        bars = ax3.barh(feat_imp_df["Sensor Feature"], feat_imp_df["Importance"] * 100, color='#2e7d32', height=0.6)
        ax3.set_title("Random Forest Relative Feature Importance (%)", fontsize=11, fontweight='bold', pad=12)
        ax3.set_xlabel("Importance (%)", fontsize=10)
        ax3.grid(axis='x', linestyle='--', alpha=0.3)
        sns.despine(top=True, right=True)
        plt.tight_layout()
        st.pyplot(fig3)

    with v_row2_col2:
        st.markdown("#### 4. Sensor Distribution Explorer")
        sensor_choice = st.selectbox(
            "Select Sensor to View Distribution:",
            FEATURE_COLS,
            format_func=lambda x: f"{FEATURE_INFO[x]['icon']} {FEATURE_INFO[x]['label']} ({FEATURE_INFO[x]['unit']})"
        )
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.histplot(
            df[sensor_choice],
            kde=True,
            color='#1b5e20',
            bins=20,
            ax=ax4
        )
        ax4.set_title(f"Distribution of {FEATURE_INFO[sensor_choice]['label']}", fontsize=11, fontweight='bold', pad=12)
        ax4.set_xlabel(f"{FEATURE_INFO[sensor_choice]['label']} ({FEATURE_INFO[sensor_choice]['unit']})", fontsize=10)
        ax4.set_ylabel("Frequency", fontsize=10)
        ax4.grid(axis='y', linestyle='--', alpha=0.3)
        sns.despine(top=True, right=True)
        plt.tight_layout()
        st.pyplot(fig4)


# =============================================================================
# TAB 3: DATASET EXPLORER
# =============================================================================
with tab3:
    st.markdown("### 📋 Smart Farming Dataset Explorer")
    st.caption(f"Inspecting raw dataset records from `{CSV_FILENAME}`")

    # Filter Controls
    f_col1, f_col2 = st.columns([1, 3])
    with f_col1:
        condition_filter = st.multiselect(
            "Filter by Condition:",
            options=list(df[TARGET_COL].unique()),
            default=list(df[TARGET_COL].unique())
        )
    with f_col2:
        search_kw = st.text_input("Search dataset (Farm ID, Region, Crop Type):", placeholder="e.g. Wheat, FARM0001, North India")

    filtered_df = df[df[TARGET_COL].isin(condition_filter)]
    if search_kw:
        mask = filtered_df.astype(str).apply(lambda row: row.str.contains(search_kw, case=False).any(), axis=1)
        filtered_df = filtered_df[mask]

    st.dataframe(filtered_df, use_container_width=True)
    st.caption(f"Showing {len(filtered_df)} of {len(df)} total records.")

    st.markdown("#### 📐 Statistical Summary of Numerical Sensor Features")
    st.dataframe(df[FEATURE_COLS].describe().round(2), use_container_width=True)


# =============================================================================
# TAB 4: COLLEGE MINI-PROJECT CONTENT & ARCHITECTURE
# =============================================================================
with tab4:
    st.markdown("### 🎓 Academic Mini-Project Documentation")
    
    st.markdown("""
    #### 🎯 AIM:
    > *"To develop a machine learning-based Smart Farming Sensor Analysis System that detects agricultural operating conditions using farming sensor and environmental data."*
    
    #### 📝 PROBLEM STATEMENT (DEFINE):
    > *"How might we develop a simple machine learning application that uses farming sensor data to identify agricultural operating conditions and provide an understandable result to the user?"*

    #### ⚠️ EXISTING PROBLEMS:
    1. **Sensor Complexity:** Farmers find raw numerical sensor data (NDVI, soil moisture percentages, pH values) difficult to interpret without expert agronomic knowledge.
    2. **Time-Consuming Manual Monitoring:** Manual inspection of large acreage takes excessive time and delays essential disease treatments.
    3. **Environmental Volatility:** Microclimate variations rapidly alter operating safety thresholds.
    4. **Lack of Accessible AI Decision Tools:** Most AI platforms require complex cloud infrastructure.
    
    #### 🎯 OBJECTIVES:
    1. Collect farming sensor and environmental data from the Kaggle dataset.
    2. Preprocess data (clean missing values, remove duplicates, select 8 sensor attributes).
    3. Train a multi-class Random Forest Classifier.
    4. Detect the operational condition (`Healthy`, `Mild`, `Moderate`, `Severe`).
    5. Deliver real-time recommendations through an interactive Streamlit UI.

    #### ⚡ 5-STEP MACHINE LEARNING ALGORITHM:
    1. **Read Dataset:** Load `Smart_Farming_Crop_Yield_2024.csv` via pandas.
    2. **Prepare Sensor Features:** Isolate 8 numerical sensor variables and clean target labels.
    3. **Train Model:** Split into 80% train and 20% test subsets; fit `RandomForestClassifier`.
    4. **Predict Condition:** Feed user-entered sensor inputs into the trained model.
    5. **Display Advisory:** Render the detected condition, confidence score, and action plan in the web UI.
    """)


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center style='color: #558b2f; font-size: 0.9rem; font-weight: 500;'>"
    "🌱 Smart Farming Sensor Analysis • Built with Python, Scikit-Learn & Streamlit"
    "</center>",
    unsafe_allow_html=True
)
