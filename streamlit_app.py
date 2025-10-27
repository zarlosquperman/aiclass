#streamlit_app.py

import streamlit as st
from fastai.vision.all import *
from PIL import Image, ImageOps
import gdown
import os

# --- 1. 페이지 기본 설정 ---
st.set_page_config(
    page_title="Fastai 이미지 분류기",
    page_icon="🤖",
)

# --- 2. 커스텀 CSS ---
st.markdown("""
<style>
h1 {
    color: #1E88E5;
    text-align: center;
    font-weight: bold;
}
.stFileUploader {
    border: 2px dashed #1E88E5;
    border-radius: 10px;
    padding: 15px;
    background-color: #f5fafe;
}
.prediction-box {
    background-color: #E3F2FD;
    border: 2px solid #1E88E5;
    border-radius: 10px;
    padding: 25px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.prediction-box h2 {
    color: #0D47A1;
    margin: 0;
    font-size: 2.0rem;
}
.prob-card {
    background-color: #FFFFFF;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    transition: transform 0.2s ease;
}
.prob-card:hover { transform: translateY(-3px); }
.prob-label {
    font-weight: bold;
    font-size: 1.05rem;
    color: #333;
}
.prob-bar-bg {
    background-color: #E0E0E0;
    border-radius: 5px;
    width: 100%;
    height: 22px;
    overflow: hidden;
}
.prob-bar-fg {
    background-color: #4CAF50;
    height: 100%;
    border-radius: 5px 0 0 5px;
    text-align: right;
    padding-right: 8px;
    color: white;
    font-weight: bold;
    line-height: 22px;
    transition: width 0.5s ease-in-out;
}
.prob-bar-fg.highlight { background-color: #FF6F00; }
</style>
""", unsafe_allow_html=True)

# --- 3. 모델 로드 ---
file_id = '19dS6rAzHlGekODz1l2F020D9XMlhNDYS'
model_path = 'model.pkl'

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
    # CPU 환경 강제 로드
    learner = load_learner(output_path, cpu=True)
    return learner

with st.spinner("🤖 AI 모델을 불러오는 중입니다. 잠시만 기다려주세요..."):
    learner = load_model_from_drive(file_id, model_path)

st.success("✅ 모델 로드가 완료되었습니다!")

labels = learner.dls.vocab
st.title("이미지 분류기 (Fastai)")
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# --- 4. 업로드 + 레이아웃(1행 2열) ---
uploaded_file = st.file_uploader(
    "분류할 이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
    type=["jpg", "png", "jpeg", "webp", "tiff"]
)

if uploaded_file is not None:
    # 1행 2열 레이아웃
    col1, col2 = st.columns([1, 1])

    # 이미지 로드: EXIF 자동 회전 + RGB 강제
    try:
        pil_img = Image.open(uploaded_file)
        pil_img = ImageOps.exif_transpose(pil_img)  # EXIF 회전 보정
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")        # RGBA/L 등 → RGB
    except Exception as e:
        st.error(f"이미지 열기/전처리 중 오류 발생: {e}")
        st.stop()

    with col1:
        # use_container_width로 경고 해결
        st.image(pil_img, caption="업로드된 이미지", use_container_width=True)

    # fastai 입력 객체 생성
    try:
        img = PILImage.create(pil_img)  # PIL Image 객체 직접 전달
    except Exception as e:
        st.error(f"fastai 이미지 변환 중 오류 발생: {e}")
        st.stop()

    with st.spinner("🧠 이미지를 분석 중입니다..."):
        prediction, pred_idx, probs = learner.predict(img)

    with col1:
        # 좌측: 예측 결과 박스
        st.markdown(f"""
        <div class="prediction-box">
            <span style="font-size: 1.0rem; color: #555;">예측 결과:</span>
            <h2>{prediction}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 우측: 상세 확률 막대
        st.markdown("<h3>상세 예측 확률:</h3>", unsafe_allow_html=True)

        # 확률 내림차순 정렬
        prob_list = sorted(
            [(lbl, float(probs[i])) for i, lbl in enumerate(labels)],
            key=lambda x: x[1],
            reverse=True
        )

        for label, prob in prob_list:
            highlight_class = "highlight" if label == str(prediction) else ""
            prob_percent = prob * 100.0

            st.markdown(f"""
            <div class="prob-card">
                <span class="prob-label">{label}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fg {highlight_class}" style="width: {prob_percent:.4f}%;">
                        {prob_percent:.2f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("이미지를 업로드하면 AI가 분석을 시작합니다.")

