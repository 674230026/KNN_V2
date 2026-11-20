import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart KNN Studio • AI Classification",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Asset Helper
# ---------------------------------------------------------
CURRENT_DIR = Path(__file__).parent
DEV_IMG_PATH = CURRENT_DIR / "img" / "developer.jpg"
DATA_PATH = CURRENT_DIR / "data" / "Heart3.csv"
NORMAL_IMG_PATH = CURRENT_DIR / "img" / "none brain.jpg"
RISK_IMG_PATH = CURRENT_DIR / "img" / "boken heart.jpg"

def load_img_base64(path):
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

dev_b64 = load_img_base64(DEV_IMG_PATH)
normal_b64 = load_img_base64(NORMAL_IMG_PATH)
risk_b64 = load_img_base64(RISK_IMG_PATH)

# ---------------------------------------------------------
# UI / UX Pro Max - Medical Tech Theme Styles
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --primary: #0284c7;
    --primary-dark: #0369a1;
    --primary-light: #e0f2fe;
    --secondary: #0d9488;
    --secondary-light: #ccfbf1;
    --success: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    --card-bg: #ffffff;
    --surface-bg: #f8fafc;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    --border-subtle: #e2e8f0;
    --border-accent: #cbd5e1;
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --shadow-soft: 0 4px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03);
    --shadow-hover: 0 10px 25px -4px rgba(2, 132, 199, 0.12), 0 4px 10px -2px rgba(2, 132, 199, 0.06);
}

html, body, [class*="css"] {
    font-family: 'Prompt', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    color: var(--text-primary);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
    max-width: 1320px;
}

/* Sidebar Custom Styling */
div[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
div[data-testid="stSidebar"] * {
    color: #e2e8f0;
}
div[data-testid="stSidebar"] .stSlider > label,
div[data-testid="stSidebar"] .stSelectbox > label {
    font-weight: 500;
    color: #cbd5e1 !important;
}

/* Sidebar Developer Profile Card */
.sidebar-dev-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 1.1rem;
    text-align: center;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
}
.sidebar-dev-img {
    width: 82px;
    height: 82px;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid #38bdf8;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    margin: 0 auto 0.6rem auto;
    display: block;
}
.sidebar-dev-name {
    font-size: 1rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0;
}
.sidebar-dev-role {
    font-size: 0.78rem;
    color: #38bdf8;
    font-weight: 500;
    margin-top: 0.15rem;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399 !important;
    font-size: 0.72rem;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    margin-top: 0.5rem;
    font-weight: 500;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    background-color: #34d399;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #34d399;
}

/* Hero Header */
.hero-card {
    background: linear-gradient(120deg, #091e3a 0%, #0d3b66 45%, #055e78 100%);
    border-radius: var(--radius-lg);
    padding: 2.2rem 2.4rem;
    color: #ffffff;
    box-shadow: 0 12px 30px rgba(9, 30, 58, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.12);
    position: relative;
    overflow: hidden;
    margin-bottom: 1.6rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56, 189, 248, 0.18);
    border: 1px solid rgba(56, 189, 248, 0.4);
    color: #7dd3fc;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 0.75rem;
    letter-spacing: 0.5px;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 0.45rem 0;
    color: #ffffff;
    line-height: 1.25;
}
.hero-subtitle {
    font-size: 1rem;
    color: #cbd5e1;
    margin: 0;
    max-width: 850px;
    line-height: 1.6;
}

/* Metric KPI Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.kpi-card {
    background: #ffffff;
    border-radius: var(--radius-md);
    padding: 1.15rem 1.3rem;
    border: 1px solid var(--border-subtle);
    box-shadow: var(--shadow-soft);
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
    border-color: #bae6fd;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0284c7, #0d9488);
}
.kpi-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.kpi-val {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0f172a;
    font-family: 'JetBrains Mono', 'Prompt', monospace;
    letter-spacing: -0.5px;
}
.kpi-sub {
    font-size: 0.76rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* Prediction Result Cards */
.result-card {
    border-radius: var(--radius-lg);
    padding: 1.8rem;
    box-shadow: var(--shadow-soft);
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    border: 1px solid transparent;
    transition: all 0.3s ease;
}
.result-safe {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border-color: #a7f3d0;
    color: #064e3b;
}
.result-danger {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-color: #fecaca;
    color: #7f1d1d;
}

/* Section Header */
.section-title {
    font-size: 1.22rem;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 1rem 0 0.8rem 0;
}
.section-badge {
    background: var(--primary-light);
    color: var(--primary-dark);
    font-size: 0.72rem;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
    font-weight: 600;
}

/* Card Container */
.content-box {
    background: #ffffff;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    box-shadow: var(--shadow-soft);
    margin-bottom: 1.4rem;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    font-family: 'Prompt', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.4rem !important;
    border-radius: 10px 10px 0 0 !important;
}

/* Form Submit Button */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: white !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45) !important;
}

/* Developer Showcase Card */
.dev-profile-showcase {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: var(--shadow-soft);
    display: flex;
    gap: 2rem;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.dev-showcase-avatar {
    width: 160px;
    height: 160px;
    border-radius: 20px;
    object-fit: cover;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);
    border: 4px solid #ffffff;
    outline: 2px solid #0284c7;
}
.dev-meta-tag {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    font-size: 0.8rem;
    padding: 0.25rem 0.75rem;
    border-radius: 8px;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Model Setup
# ---------------------------------------------------------
FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]
NUMERIC = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ชุดข้อมูลที่: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + ["HeartDisease"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"ไม่พบคอลัมน์ที่จำเป็น: {missing}")
    return data[required].copy()

@st.cache_resource
def train_model(k_val, test_ratio, metric_name="minkowski", weights_mode="distance"):
    data = load_data()
    X = data[FEATURES]
    y = data["HeartDisease"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )
    
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ])
    
    p_val = 1 if metric_name == "manhattan" else 2
    
    model = Pipeline([
        ("preprocess", prep),
        ("knn", KNeighborsClassifier(
            n_neighbors=k_val,
            weights=weights_mode,
            metric=metric_name,
            p=p_val
        ))
    ])
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "matrix": confusion_matrix(y_test, pred),
        "y_test": y_test,
        "pred": pred,
        "train_count": len(X_train),
        "test_count": len(X_test)
    }
    return model, metrics

# Load Data
try:
    df = load_data()
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    # 1. Developer Card with Uploaded Photo
    if dev_b64:
        st.markdown(f"""
        <div class="sidebar-dev-card">
            <img src="data:image/jpeg;base64,{dev_b64}" class="sidebar-dev-img" alt="ผู้พัฒนาโครงงาน" />
            <p class="sidebar-dev-name">ผู้พัฒนาโครงงาน</p>
            <p class="sidebar-dev-role">Data Science & AI Studio</p>
            <div class="status-pill">
                <span class="pulse-dot"></span> ระบบพร้อมทำงาน
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif DEV_IMG_PATH.exists():
        st.image(str(DEV_IMG_PATH), caption="ผู้พัฒนาโครงงาน", use_container_width=True)

    st.markdown("### ⚙️ ปรับแต่งพารามิเตอร์โมเดล")
    
    k_selected = st.slider(
        "จำนวนเพื่อนบ้าน (K-Neighbors)",
        min_value=1,
        max_value=25,
        value=5,
        step=2,
        help="จำนวนจุดข้อมูลเพื่อนบ้านที่ใกล้ที่สุดที่นำมาพิจารณาในการจำแนก"
    )
    
    test_percent = st.slider(
        "สัดส่วนชุดข้อมูลทดสอบ (Test Size)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        format="%d%%",
        help="เปอร์เซ็นต์ของชุดข้อมูลที่สุ่มแบ่งไว้สำหรับทดสอบวัดประสิทธิภาพ"
    )

    with st.expander("🛠️ ตัวเลือกเพิ่มเติมของอัลกอริทึม"):
        weight_choice = st.selectbox(
            "น้ำหนักเพื่อนบ้าน (Weights)",
            options=["distance", "uniform"],
            format_func=lambda x: "ถ่วงน้ำหนักตามระยะห่าง (Distance)" if x == "distance" else "น้ำหนักเท่ากันทั้งหมด (Uniform)"
        )
        metric_choice = st.selectbox(
            "ฟังก์ชันวัดระยะ (Distance Metric)",
            options=["minkowski", "manhattan"],
            format_func=lambda x: "Euclidean (Minkowski p=2)" if x == "minkowski" else "Manhattan (L1 norm)"
        )
    
    st.markdown("---")
    st.caption("💡 **คำแนะนำด้าน UX:** อัลกอริทึม KNN มีการแปลงคุณลักษณะตัวเลขด้วย `StandardScaler` และตัวแปรหมวดหมู่ด้วย `OneHotEncoder` อัตโนมัติ")

# Train model based on sidebar selections
model, result = train_model(
    k_val=k_selected,
    test_ratio=test_percent / 100.0,
    metric_name=metric_choice,
    weights_mode=weight_choice
)

# ---------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------
col_hero_text, col_hero_badge = st.columns([4, 1])
with col_hero_text:
    st.markdown("""
    <div class="hero-card">
        <div class="hero-badge">✨ K-NEAREST NEIGHBORS • HEART STUDIO V2.0</div>
        <h1 class="hero-title">ระบบจำแนกและประเมินความเสี่ยงโรคหัวใจ</h1>
        <p class="hero-subtitle">
            วิเคราะห์และคาดการณ์ความเสี่ยงโรคหัวใจด้วย Machine Learning (KNN) บนชุดข้อมูล Heart3.csv 
            พร้อมสถิติภาพรวมตัวชี้วัดสรีรวิทยา และการทดสอบโมเดลแบบโต้ตอบ
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------
tab_pred, tab_data, tab_eval, tab_about = st.tabs([
    "🩺 ทดลองทำนายผลรายบุคคล",
    "📊 ภาพรวม & การสำรวจข้อมูล",
    "⚡ ประสิทธิภาพของโมเดล",
    "👨‍💻 ผู้พัฒนาโครงงาน & สถาปัตยกรรม"
])

# ---------------------------------------------------------
# TAB 1: PREDICTION STUDIO
# ---------------------------------------------------------
with tab_pred:
    st.markdown('<div class="section-title">🩺 กรอกข้อมูลคุณลักษณะเพื่อทำนายผล</div>', unsafe_allow_html=True)
    st.write("กรอกข้อมูลผลการตรวจวัดสุขภาพทั้ง 11 ปัจจัย ระบบจะคำนวณและประมวลผลผ่านโมเดล KNN ทันที")
    
    with st.form("interactive_prediction_form"):
        col_group1, col_group2, col_group3 = st.columns(3)
        
        with col_group1:
            st.markdown("##### 👤 ข้อมูลพื้นฐาน & สรีระ")
            age_in = st.number_input(
                "Age — อายุ (ปี)",
                min_value=1,
                max_value=120,
                value=54,
                help="อายุของผู้เข้ารับการประเมิน"
            )
            sex_in = st.selectbox(
                "Sex — เพศ",
                options=[1, 0],
                format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)"
            )
            resting_bp_in = st.number_input(
                "RestingBP — ความดันโลหิตขณะพัก (mm Hg)",
                min_value=0,
                max_value=300,
                value=130,
                help="ความดันโลหิตขณะนั่งพัก ปกติอยู่ช่วง 90-120 mm Hg"
            )
            max_hr_in = st.number_input(
                "MaxHR — อัตราการเต้นหัวใจสูงสุด (bpm)",
                min_value=40,
                max_value=250,
                value=145,
                help="อัตราการเต้นหัวใจสูงสุดที่วัดได้"
            )
            
        with col_group2:
            st.markdown("##### 🧪 สารเคมีในเลือด & อาการเจ็บหน้าอก")
            chol_in = st.number_input(
                "Cholesterol — คอเลสเตอรอลในซีรัม (mg/dl)",
                min_value=0,
                max_value=1000,
                value=220,
                help="ระดับคอเลสเตอรอลในกระแสเลือด ปกติ < 200 mg/dl"
            )
            fasting_in = st.selectbox(
                "FastingBS — น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x: "ปกติ / ไม่เกินเกณฑ์ (0)" if x == 0 else "สูงกว่า 120 mg/dl (1)"
            )
            chest_in = st.selectbox(
                "ChestPainType — ลักษณะอาการเจ็บหน้าอก",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 — Typical Angina (เจ็บตามแบบแผน)",
                    2: "2 — Atypical Angina (เจ็บไม่ตรงแบบแผน)",
                    3: "3 — Non-Anginal Pain (เจ็บที่มิใช่โรคหัวใจ)",
                    4: "4 — Asymptomatic (ไม่มีอาการเจ็บ)"
                }[x]
            )
            angina_in = st.selectbox(
                "ExerciseAngina — แน่นหน้าอกเมื่อออกกำลังกาย",
                options=[0, 1],
                format_func=lambda x: "ไม่มีอาการ (0)" if x == 0 else "มีอาการแน่นหน้าอก (1)"
            )
            
        with col_group3:
            st.markdown("##### 📉 ผลการตรวจคลื่นไฟฟ้าหัวใจ (ECG)")
            ecg_in = st.selectbox(
                "RestingECG — ผลคลื่นไฟฟ้าหัวใจขณะพัก",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1 — Normal (ปกติ)",
                    2: "2 — ST-T Wave Abnormality (คลื่นผิดปกติ)",
                    3: "3 — Left Ventricular Hypertrophy (ผนังหัวใจหนา)"
                }[x]
            )
            oldpeak_in = st.number_input(
                "Oldpeak — การลดระดับของ ST",
                min_value=-5.0,
                max_value=15.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                help="ค่า ST Depression ที่วัดจากการตรวจสมรรถภาพหัวใจขณะออกกำลังกาย"
            )
            slope_in = st.selectbox(
                "ST_Slope — ความลาดเอียงของช่วง ST",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1 — Upsloping (ลาดขึ้น)",
                    2: "2 — Flat (แนวราบ)",
                    3: "3 — Downsloping (ลาดลง)"
                }[x]
            )
            
        submitted = st.form_submit_button("🔍 ทำนายผลประเมินทันที", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "Age": age_in, "Sex": sex_in, "ChestPainType": chest_in,
            "RestingBP": resting_bp_in, "Cholesterol": chol_in, "FastingBS": fasting_in,
            "RestingECG": ecg_in, "MaxHR": max_hr_in, "ExerciseAngina": angina_in,
            "Oldpeak": oldpeak_in, "ST_Slope": slope_in
        }], columns=FEATURES)
        
        pred_val = int(model.predict(input_data)[0])
        probs = model.predict_proba(input_data)[0]
        classes_list = list(model.named_steps["knn"].classes_)
        class_idx = classes_list.index(pred_val)
        confidence = float(probs[class_idx])
        
        st.markdown('<div class="section-title">📋 ผลการประเมินจากโมเดล</div>', unsafe_allow_html=True)
        
        col_res_banner, col_res_img = st.columns([3, 1])
        
        with col_res_banner:
            if pred_val == 1:
                st.markdown(f"""
                <div class="result-card result-danger">
                    <h3 style="margin-top:0; color:#991b1b; display:flex; align-items:center; gap:8px;">
                        ⚠️ ผลการทำนาย: พบความเสี่ยงเป็นโรคหัวใจ (Class 1)
                    </h3>
                    <p style="font-size:1.05rem; line-height:1.6; margin-bottom: 0.8rem;">
                        โมเดล KNN จัดกลุ่มข้อมูลนี้อยู่ในกลุ่ม <b>ผู้มีความเสี่ยงหรือพบรอยโรค</b> 
                        โดยมีระดับความเชื่อมั่นของโมเดลที่ <b>{confidence:.1%}</b> 
                        (พิจารณาจากกลุ่มเพื่อนบ้านที่ใกล้เคียงที่สุดจำนวน {k_selected} ตัวอย่าง)
                    </p>
                    <div style="background:rgba(255,255,255,0.7); border-radius:10px; padding:0.6rem 1rem; font-size:0.88rem; color:#7f1d1d;">
                        <b>คำแนะนำเบื้องต้น:</b> ควรพบแพทย์ผู้เชี่ยวชาญด้านโรคหัวใจเพื่อตรวจวินิจฉัยเชิงลึกด้วยเครื่องมือทางการแพทย์ต่อไป
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-safe">
                    <h3 style="margin-top:0; color:#065f46; display:flex; align-items:center; gap:8px;">
                        ✅ ผลการทำนาย: ไม่พบความเสี่ยงโรคหัวใจ (Class 0 - ปกติ)
                    </h3>
                    <p style="font-size:1.05rem; line-height:1.6; margin-bottom: 0.8rem;">
                        โมเดล KNN จัดกลุ่มข้อมูลนี้อยู่ในกลุ่ม <b>ผลตรวจสุขภาพหัวใจปกติ</b> 
                        โดยมีระดับความเชื่อมั่นของโมเดลที่ <b>{confidence:.1%}</b> 
                        (พิจารณาจากกลุ่มเพื่อนบ้านที่ใกล้เคียงที่สุดจำนวน {k_selected} ตัวอย่าง)
                    </p>
                    <div style="background:rgba(255,255,255,0.7); border-radius:10px; padding:0.6rem 1rem; font-size:0.88rem; color:#064e3b;">
                        <b>คำแนะนำเบื้องต้น:</b> หมั่นรักษาระดับความดันและคอเลสเตอรอลให้อยู่ในเกณฑ์มาตรฐาน และออกกำลังกายสม่ำเสมอ
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Progress Bar showing confidence
            st.write(f"**ระดับความเชื่อมั่นของโมเดล (Model Probability):** {confidence:.1%}")
            st.progress(min(max(confidence, 0.0), 1.0))
            
        with col_res_img:
            # Display medical reference image
            if pred_val == 1 and RISK_IMG_PATH.exists():
                st.image(str(RISK_IMG_PATH), caption="ภาพจำลอง: สภาวะเสี่ยงโรคหัวใจ", use_container_width=True)
            elif pred_val == 0 and NORMAL_IMG_PATH.exists():
                st.image(str(NORMAL_IMG_PATH), caption="ภาพจำลอง: สภาวะหัวใจปกติ", use_container_width=True)

        with st.expander("🔍 ดูข้อมูลอินพุตและโครงสร้างเวกเตอร์ที่คุณกรอก"):
            st.dataframe(input_data, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: DATASET OVERVIEW & EXPLORATION
# ---------------------------------------------------------
with tab_data:
    st.markdown('<div class="section-title">📊 ภาพรวมชุดข้อมูล Heart3.csv</div>', unsafe_allow_html=True)
    
    total_rows = len(df)
    pos_count = int((df["HeartDisease"] == 1).sum())
    neg_count = int((df["HeartDisease"] == 0).sum())
    pos_pct = (pos_count / total_rows) * 100
    
    # 4 KPI Cards
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">จำนวนข้อมูลทั้งหมด <span>📁</span></div>
            <div class="kpi-val">{total_rows:,}</div>
            <div class="kpi-sub">แถวข้อมูลที่นำมาวิเคราะห์</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">คุณลักษณะที่ใช้ <span>🧬</span></div>
            <div class="kpi-val">{len(FEATURES)}</div>
            <div class="kpi-sub">ตัวแปรตรวจวัดสุขภาพ</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">พบโรคหัวใจ (Class 1) <span>🔴</span></div>
            <div class="kpi-val">{pos_count:,}</div>
            <div class="kpi-sub">คิดเป็น {pos_pct:.1f}% ของข้อมูล</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">ปกติ (Class 0) <span>🟢</span></div>
            <div class="kpi-val">{neg_count:,}</div>
            <div class="kpi-sub">คิดเป็น {100-pos_pct:.1f}% ของข้อมูล</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visual Chart for Physiological Means
    col_chart, col_stats = st.columns([3, 2])
    with col_chart:
        st.markdown("##### 📈 ค่าเฉลี่ยตัวชี้วัดสรีรวิทยาที่สำคัญ")
        means = {
            "RestingBP\n(ความดันขณะพัก)": float(df["RestingBP"].mean()),
            "Cholesterol\n(คอเลสเตอรอล)": float(df["Cholesterol"].mean()),
            "MaxHR\n(หัวใจเต้นสูงสุด)": float(df["MaxHR"].mean()),
            "Age\n(อายุเฉลี่ย)": float(df["Age"].mean())
        }
        
        fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=100)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')
        
        bars = ax.bar(
            means.keys(),
            means.values(),
            color=['#0284c7', '#0d9488', '#38bdf8', '#64748b'],
            width=0.48,
            edgecolor='#e2e8f0',
            linewidth=1.2,
            zorder=3
        )
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(colors='#475569', labelsize=8.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='600', color='#0f172a')
                        
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_stats:
        st.markdown("##### 🩺 ข้อมูลการกระจายตัว")
        st.write(df[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]].describe().T[["mean", "std", "min", "max"]])
        
        # Zero-value alert if any
        zero_chol = int((df['Cholesterol'] == 0).sum())
        zero_bp = int((df['RestingBP'] == 0).sum())
        if zero_chol > 0 or zero_bp > 0:
            st.info(f"ℹ️ **ข้อสังเกตข้อมูล:** พบค่า 0 ใน Cholesterol ({zero_chol} แถว) และ RestingBP ({zero_bp} แถว) ในชุดข้อมูลเดิม ซึ่งอาจเป็นค่า Missing Value ในเชิงปฏิบัติ")

    with st.expander("📋 เปิดดูตารางชุดข้อมูล Heart3.csv (10 แถวแรก)"):
        st.dataframe(df.head(15), use_container_width=True)

# ---------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & METRICS
# ---------------------------------------------------------
with tab_eval:
    st.markdown('<div class="section-title">⚡ การประเมินและวัดผลโมเดล KNN</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Accuracy (ความแม่นยำรวม)</div>
            <div class="kpi-val">{result['accuracy']:.1%}</div>
            <div class="kpi-sub">สัดส่วนการทำนายถูกต้องทั้งหมด</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Precision (Class 1)</div>
            <div class="kpi-val">{result['precision']:.1%}</div>
            <div class="kpi-sub">ความแม่นยำในการระบุผู้มีโรค</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Recall (Class 1)</div>
            <div class="kpi-val">{result['recall']:.1%}</div>
            <div class="kpi-sub">ความสามารถในการจับผู้ป่วยได้ครบ</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">F1-Score</div>
            <div class="kpi-val">{result['f1']:.1%}</div>
            <div class="kpi-sub">ค่าเฉลี่ยฮาร์มอนิกของ P & R</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"📌 การทดสอบใช้การแบ่งข้อมูลแบบ Stratified Train/Test Split (Train {result['train_count']} ตัวอย่าง, Test {result['test_count']} ตัวอย่าง) • K = {k_selected}")
    
    col_cm, col_report = st.columns([1, 1])
    with col_cm:
        st.markdown("##### 🎯 Confusion Matrix")
        cm_df = pd.DataFrame(
            result["matrix"],
            index=["ความเป็นจริง: ปกติ (0)", "ความเป็นจริง: เสี่ยงโรค (1)"],
            columns=["ทำนาย: ปกติ (0)", "ทำนาย: เสี่ยงโรค (1)"]
        )
        st.dataframe(cm_df, use_container_width=True)
        
        tn, fp, fn, tp = result["matrix"].ravel()
        st.caption(f"• **True Negatives:** {tn} | • **False Positives:** {fp} | • **False Negatives:** {fn} | • **True Positives:** {tp}")

    with col_report:
        st.markdown("##### 📑 Classification Report")
        report_text = classification_report(
            result["y_test"],
            result["pred"],
            target_names=["ปกติ (Class 0)", "เสี่ยงโรคหัวใจ (Class 1)"],
            zero_division=0
        )
        st.code(report_text, language="text")

# ---------------------------------------------------------
# TAB 4: DEVELOPER PROFILE & ALGORITHM INFO
# ---------------------------------------------------------
with tab_about:
    st.markdown('<div class="section-title">👨‍💻 ข้อมูลผู้พัฒนา & สถาปัตยกรรมระบบ</div>', unsafe_allow_html=True)
    
    col_dev_avatar, col_dev_details = st.columns([1, 2])
    
    with col_dev_avatar:
        if DEV_IMG_PATH.exists():
            st.image(str(DEV_IMG_PATH), caption="ผู้พัฒนาโครงงาน / Project Developer", use_container_width=True)
        else:
            st.info("ไม่พบไฟล์รูปภาพผู้พัฒนา")
            
    with col_dev_details:
        st.markdown("""
        ### โครงงานระบบจำแนกข้อมูลโรคหัวใจ (Heart Disease Classification)
        **ภาควิชา/กลุ่มการเรียนรู้:** วิทยาการข้อมูลและการเรียนรู้ของเครื่อง (Machine Learning & Data Science)
        
        #### 🎯 วัตถุประสงค์ของโครงงาน:
        1. ศึกษาและประยุกต์ใช้อัลกอริทึม **K-Nearest Neighbors (KNN)** ในการจำแนกข้อมูลทางการแพทย์
        2. พัฒนาเว็บแอปพลิเคชันเชิงโต้ตอบด้วย **Streamlit** ที่มี UI/UX ทันสมัย ใช้งานง่าย และเข้าถึงได้ทุกคน
        3. ศึกษาผลกระทบของการปรับแต่งค่าพารามิเตอร์ $K$ และฟังก์ชันวัดระยะทางต่อความแม่นยำของโมเดล
        
        #### 🛠️ เทคโนโลยีที่นำมาใช้ (Tech Stack):
        """)
        st.markdown("""
        <div style="margin-top: 0.5rem;">
            <span class="dev-meta-tag">🐍 Python 3.12</span>
            <span class="dev-meta-tag">⚡ Streamlit</span>
            <span class="dev-meta-tag">🔬 Scikit-Learn</span>
            <span class="dev-meta-tag">🐼 Pandas</span>
            <span class="dev-meta-tag">🔢 NumPy</span>
            <span class="dev-meta-tag">📊 Matplotlib</span>
            <span class="dev-meta-tag">🎨 UI/UX Pro Max</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ---
        #### 📐 หลักการทำงานของ K-Nearest Neighbors (KNN):
        KNN เป็นอัลกอริทึมแบบ **Instance-Based Learning** ที่ทำงานโดย:
        1. ทำการปรับสเกลข้อมูลเชิงตัวเลขให้มีค่าเฉลี่ยเป็น 0 และความแปรปรวนเป็น 1 (StandardScaler)
        2. ทำการแปลงข้อมูลประเภทหมวดหมู่ (One-Hot Encoding)
        3. เมื่อมีข้อมูลใหม่เข้ามา จะคำนวณระยะห่าง (Distance เช่น Euclidean Distance) ไปยังทุกจุดในชุดข้อมูล
        4. คัดเลือกจุดที่ใกล้ที่สุดจำนวน $K$ จุด แล้วทำการโหวต (Majority Vote) หรือถ่วงน้ำหนักตามส่วนกลับของระยะทาง
        """)

# ---------------------------------------------------------
# FOOTER & CLINICAL DISCLAIMER
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
    <b>⚠️ ข้อสงวนสิทธิ์ทางการแพทย์ (Clinical Disclaimer):</b> 
    ระบบนี้พัฒนาขึ้นเพื่อการศึกษาและการวิจัยทางด้านวิทยาการคอมพิวเตอร์และการเรียนรู้ของเครื่องเท่านั้น 
    ผลการทำนายไม่สามารถนำมาใช้แทนการวินิจฉัยทางการแพทย์หรือการตัดสินใจในการรักษาจริงได้ 
    <br>© 2026 Heart KNN Classification Studio • พัฒนาด้วย Streamlit & Python
</div>
""", unsafe_allow_html=True)
