from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคการจำแนกข้อมูลโรคหัวใจ (Heart Disease)")
#st.image("./img/kairung022.jpg")
col1, col2 = st.columns(2)

with col1:
   st.header("ปกติ (Normal)")
   st.image("./img/none brain.jpg")

with col2:
   st.header("เสี่ยงโรคหัวใจ (Heart Disease)")
   st.image("./img/boken heart.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

dt1 = dt['RestingBP'].mean()
dt2 = dt['Cholesterol'].mean()
dt3 = dt['MaxHR'].mean()
dt4 = dt['Age'].mean()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["RestingBP (เฉลี่ย)", "Cholesterol (เฉลี่ย)", "MaxHR (เฉลี่ย)", "Age (เฉลี่ย)"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

age = st.number_input("กรุณาเลือกข้อมูล Age (อายุ)")
sex = st.number_input("กรุณาเลือกข้อมูล Sex (เพศ: 0=หญิง, 1=ชาย)")
ct = st.number_input("กรุณาเลือกข้อมูล ChestPainType (ประเภทอาการเจ็บหน้าอก 1-4)")
bp = st.number_input("กรุณาเลือกข้อมูล RestingBP (ความดันโลหิตขณะพัก)")
cho = st.number_input("กรุณาเลือกข้อมูล Cholesterol (คอเลสเตอรอล)")
bs = st.number_input("กรุณาเลือกข้อมูล FastingBS (น้ำตาลในเลือด > 120 mg/dl: 0 หรือ 1)")
ecg = st.number_input("กรุณาเลือกข้อมูล RestingECG (ผลคลื่นไฟฟ้าหัวใจ 0-2)")
hr = st.number_input("กรุณาเลือกข้อมูล MaxHR (อัตราการเต้นของหัวใจสูงสุด)")
ea = st.number_input("กรุณาเลือกข้อมูล ExerciseAngina (อาการแน่นหน้าอกขณะออกกำลังกาย 0 หรือ 1)")
op = st.number_input("กรุณาเลือกข้อมูล Oldpeak", value=0.0, step=0.1, format="%.1f")
ss = st.number_input("กรุณาเลือกข้อมูล ST_Slope (ความชันช่วง ST 1-3)")

if st.button("ทำนายผล"):
   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease 
   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  

   x_input = pd.DataFrame([[age, sex, ct, bp, cho, bs, ecg, hr, ea, op, ss]], columns=X.columns)

   out = Knn_model.predict(x_input)
   st.write(f"ผลลัพธ์จากโมเดล (0=ปกติ, 1=เสี่ยงโรคหัวใจ): {out[0]}")

   if out[0] == 0:
       st.success("ผลการทำนาย: ปกติ (ไร้....หัวใจ)")
       st.image("./img/none brain.jpg")
   elif out[0] == 1:
       st.error("ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ อาจะอยู่ได้ไม่นานา")
       st.image("./img/boken heart.jpg")
else:
   st.write("ไม่ทำนาย")
