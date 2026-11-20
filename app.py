import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Heart KNN • Classification Studio", page_icon="🫀", layout="wide")

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {font-family: 'Prompt', sans-serif;}
.stApp {background: linear-gradient(135deg,#f6f8fc 0%,#eef3fb 55%,#f8faff 100%);}
.block-container {padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1250px;}
.hero {padding: 1.8rem 2rem; border-radius: 24px; color: white;
       background: linear-gradient(115deg,#102a43,#176b87 58%,#27a6a1);
       box-shadow: 0 14px 35px rgba(16,42,67,.16); margin-bottom: 1.2rem;}
.hero h1 {font-size: 2.15rem; margin:0 0 .35rem 0; font-weight:700;}
.hero p {font-size:1rem; opacity:.9; margin:0;}
.metric {background:white; border:1px solid #e7edf5; border-radius:18px; padding:1rem 1.15rem;
         box-shadow:0 5px 18px rgba(20,50,80,.045);}
.metric-label {color:#64748b; font-size:.85rem;}
.metric-value {font-size:1.65rem; font-weight:700; color:#102a43;}
.section {font-size:1.25rem; font-weight:700; color:#102a43; margin:.6rem 0 .8rem;}
.small-note {color:#64748b; font-size:.85rem;}
div[data-testid="stForm"] {background:white; border:1px solid #e7edf5; border-radius:18px; padding:1rem 1.2rem;}
div[data-testid="stSidebar"] {background:#102a43;}
div[data-testid="stSidebar"] * {color:#f3f7fb;}
.stButton>button {border-radius:12px; font-weight:600; padding:.65rem 1rem;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "data/Heart3.csv"
FEATURES = ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS",
            "RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]
NUMERIC = ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + ["HeartDisease"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"ไม่พบคอลัมน์ที่จำเป็น: {missing}")
    return data[required].copy()

@st.cache_resource
def train_model(k, test_size):
    data = load_data()
    X = data[FEATURES]
    y = data["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ])
    model = Pipeline([
        ("preprocess", prep),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance", metric="minkowski", p=2))
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

try:
    df = load_data()
except Exception as e:
    st.error(f"โหลดไฟล์ Heart3.csv ไม่สำเร็จ: {e}")
    st.stop()

st.markdown("""
<div class="hero">
  <h1>🫀 Heart KNN · Classification Studio</h1>
  <p>ระบบทดลองจำแนกข้อมูลจากชุดข้อมูล Heart3.csv ด้วย K-Nearest Neighbors</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("## ⚙️ ตั้งค่าโมเดล")
k = st.sidebar.slider("จำนวนเพื่อนบ้าน (K)", min_value=1, max_value=25, value=5, step=2)
test_percent = st.sidebar.slider("สัดส่วนข้อมูลทดสอบ", 10, 40, 20, 5)
st.sidebar.markdown("---")
st.sidebar.caption("โมเดลใช้ StandardScaler กับข้อมูลตัวเลข และ One-Hot Encoding กับตัวแปรรหัสหมวดหมู่ก่อนคำนวณระยะห่าง")
st.sidebar.caption("ค่า K ควรทดลองหลายค่าเพื่อเปรียบเทียบผล ไม่ได้มีค่าที่ดีที่สุดตายตัว")

model, result = train_model(k, test_percent / 100)
n = len(df)
positive = int((df["HeartDisease"] == 1).sum())
negative = int((df["HeartDisease"] == 0).sum())

# Dataset overview
st.markdown('<div class="section">ภาพรวมชุดข้อมูล</div>', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
for col, label, value in [
    (c1,"จำนวนข้อมูลทั้งหมด",f"{n:,} แถว"),
    (c2,"จำนวนคุณลักษณะ",f"{len(FEATURES)} ตัวแปร"),
    (c3,"HeartDisease = 1",f"{positive:,} แถว"),
    (c4,"HeartDisease = 0",f"{negative:,} แถว")]:
    with col:
        st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

with st.expander("ดูตัวอย่างข้อมูลและรายละเอียดคอลัมน์"):
    st.dataframe(df.head(10), use_container_width=True)
    st.write("ชนิดข้อมูล:", df.dtypes.astype(str).to_dict())
    st.write("ค่าว่างทั้งหมด:", int(df.isna().sum().sum()))
    st.write("แถวข้อมูลซ้ำทั้งหมด:", int(df.duplicated().sum()))
    st.caption("ไฟล์มีรหัสตัวเลขในตัวแปรหมวดหมู่ แต่ไม่มีคำอธิบายความหมายของรหัสแนบมา จึงแสดงเป็นรหัสเดิมเพื่อไม่เดาความหมาย")

if (df[["RestingBP","Cholesterol"]] == 0).any().any():
    st.warning("พบค่า 0 ใน RestingBP หรือ Cholesterol ในชุดข้อมูลเดิม (Cholesterol = 0 จำนวน "
               f"{int((df['Cholesterol']==0).sum())} แถว, RestingBP = 0 จำนวน {int((df['RestingBP']==0).sum())} แถว) "
               "โปรดตรวจสอบที่มาของข้อมูล เพราะอาจเป็นค่าที่ใช้แทนข้อมูลที่ไม่มี แต่แอปนี้ยังคงใช้ค่าตามไฟล์ต้นฉบับ")

# Model evaluation
st.markdown('<div class="section">ผลการประเมินโมเดล</div>', unsafe_allow_html=True)
m1,m2,m3,m4 = st.columns(4)
for col, label, val in [
    (m1,"Accuracy",result["accuracy"]),
    (m2,"Precision (class 1)",result["precision"]),
    (m3,"Recall (class 1)",result["recall"]),
    (m4,"F1-score",result["f1"])]:
    with col:
        st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{val:.1%}</div></div>', unsafe_allow_html=True)
st.caption(f"แบ่งข้อมูลแบบ Stratified Train/Test Split • Train {result['train_count']} แถว • Test {result['test_count']} แถว • random_state = 42 • K = {k}")
cm = pd.DataFrame(result["matrix"], index=["จริง: 0","จริง: 1"], columns=["ทำนาย: 0","ทำนาย: 1"])
st.write("**Confusion Matrix**")
st.dataframe(cm, use_container_width=True)
with st.expander("รายงานผลจำแนกแต่ละคลาส"):
    st.text(classification_report(result["y_test"], result["pred"], target_names=["0 = ไม่พบโรคในชุดข้อมูล","1 = พบโรคในชุดข้อมูล"], zero_division=0))

# Prediction form
st.markdown('<div class="section">ทดลองทำนายข้อมูลใหม่</div>', unsafe_allow_html=True)
st.write("กรอกค่าคุณลักษณะทั้ง 11 ตัวแปร แล้วกดปุ่มทำนาย ระบบจะใช้โมเดล KNN ที่ฝึกจาก Heart3.csv")
with st.form("prediction_form"):
    a,b,c = st.columns(3)
    with a:
        age = st.number_input("Age — อายุ (ปี)", min_value=1, max_value=110, value=50)
        sex = st.selectbox("Sex — รหัสเพศ", options=[0,1], format_func=lambda x: f"รหัส {x}")
        chest = st.selectbox("ChestPainType — ประเภทอาการเจ็บหน้าอก (รหัส)", options=[1,2,3,4])
        resting_bp = st.number_input("RestingBP — ความดันขณะพัก", min_value=0, max_value=300, value=120)
    with b:
        cholesterol = st.number_input("Cholesterol — คอเลสเตอรอล", min_value=0, max_value=1000, value=200)
        fasting = st.selectbox("FastingBS — น้ำตาลขณะอดอาหาร (รหัส)", options=[0,1], format_func=lambda x: f"รหัส {x}")
        ecg = st.selectbox("RestingECG — ผล ECG ขณะพัก (รหัส)", options=[1,2,3])
        max_hr = st.number_input("MaxHR — อัตราการเต้นหัวใจสูงสุด", min_value=1, max_value=300, value=150)
    with c:
        angina = st.selectbox("ExerciseAngina — เจ็บหน้าอกจากการออกกำลัง (รหัส)", options=[0,1], format_func=lambda x: f"รหัส {x}")
        oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=20.0, value=0.0, step=0.1, format="%.1f")
        slope = st.selectbox("ST_Slope — ความชัน ST (รหัส)", options=[1,2,3])
        st.info("กรุณากรอกค่าจากผลตรวจหรือข้อมูลที่ได้รับอย่างถูกต้อง ไม่ควรเดาค่าทางการแพทย์")
    submitted = st.form_submit_button("🔎 ทำนายข้อมูล", use_container_width=True, type="primary")

if submitted:
    new_row = pd.DataFrame([{
        "Age":age, "Sex":sex, "ChestPainType":chest, "RestingBP":resting_bp,
        "Cholesterol":cholesterol, "FastingBS":fasting, "RestingECG":ecg,
        "MaxHR":max_hr, "ExerciseAngina":angina, "Oldpeak":oldpeak, "ST_Slope":slope
    }], columns=FEATURES)
    prediction = int(model.predict(new_row)[0])
    probabilities = model.predict_proba(new_row)[0]
    class_index = list(model.named_steps["knn"].classes_).index(prediction)
    confidence = float(probabilities[class_index])
    st.markdown("### ผลการจำแนกจากโมเดล")
    if prediction == 1:
        st.error(f"ผลทำนาย: คลาส 1 — โมเดลจัดข้อมูลนี้อยู่ในกลุ่ม HeartDisease = 1 (คะแนนความน่าจะเป็นจากโมเดล {confidence:.1%})")
    else:
        st.success(f"ผลทำนาย: คลาส 0 — โมเดลจัดข้อมูลนี้อยู่ในกลุ่ม HeartDisease = 0 (คะแนนความน่าจะเป็นจากโมเดล {confidence:.1%})")
    st.caption("คะแนนนี้เป็นค่าความน่าจะเป็นที่โมเดลคำนวณ ไม่ใช่ความเสี่ยงทางการแพทย์หรือการวินิจฉัยโรค")
    st.dataframe(new_row, use_container_width=True)

st.markdown("---")
st.markdown("""
<div class="small-note">
<b>ข้อจำกัดสำคัญ:</b> แอปนี้จัดทำเพื่อการเรียนรู้และทดลองอัลกอริทึมเท่านั้น ไม่ใช่อุปกรณ์หรือบริการวินิจฉัยโรค
ผลทำนายขึ้นกับคุณภาพและรูปแบบการเข้ารหัสของ Heart3.csv ไม่ควรใช้ตัดสินใจเรื่องการรักษาหรือสุขภาพจริง
</div>
""", unsafe_allow_html=True)
