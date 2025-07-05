import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 데이터 URL (raw 형태로 반드시 수정)
DATA_URL = "https://raw.githubusercontent.com/yeyjin123/bio-ai/main/parkinson_cleaned.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

st.title("파킨슨병 예측 인공지능 (음성 기반)")

st.markdown("""
이 웹앱은 사용자의 음성 기반 생체 지표를 활용하여 파킨슨병 여부를 예측합니다.  
입력값으로는 nhr(잡음 비율)과 rpde(음성의 복잡도)를 사용합니다.
""")

if st.checkbox("원본 데이터 보기"):
    st.dataframe(df[['nhr', 'rpde', 'parkinson_status']].head())

X = df[['nhr', 'rpde']]
y = df['parkinson_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

st.subheader("생체신호를 입력하세요:")
nhr_val = st.number_input("nhr (잡음 비율)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
rpde_val = st.number_input("rpde (비선형 지표)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

if st.button("파킨슨병 예측하기"):
    prediction = model.predict([[nhr_val, rpde_val]])[0]
    probability = model.predict_proba([[nhr_val, rpde_val]])[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ 파킨슨병일 가능성이 높습니다 (예측 확률: {probability:.2%})")
    else:
        st.success(f"✅ 정상으로 예측됩니다 (예측 확률: {probability:.2%})")
