import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown
import os # 파일 존재 여부 확인을 위해 import

# --- 1. 페이지 기본 설정 ---
st.set_page_config(
    page_title="Fastai 이미지 분류기",
    page_icon="🤖"
)

# --- 2. 커스텀 CSS ---
st.markdown("""
<style>
/* 페이지 타이틀 */
h1 {
    color: #1E88E5; /* Blue */
    text-align: center;
    font-weight: bold;
}

/* 파일 업로더 */
.stFileUploader {
    border: 2px dashed #1E88E5;
    border-radius: 10px;
    padding: 15px;
    background-color: #f5fafe;
}

/* 예측 결과 박스 */
.prediction-box {
    background-color: #E3F2FD; /* Light Blue */
    border: 2px solid #1E88E5;
    border-radius: 10px;
    padding: 25px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

.prediction-box h2 {
    color: #0D47A1; /* Dark Blue */
    margin: 0;
    font-size: 2.5rem; /* 글자 크기 키움 */
}

/* 확률 표시용 카드 스타일 */
.prob-card {
    background-color: #FFFFFF;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.08);
    transition: transform 0.2s ease;
}
.prob-card:hover {
    transform: translateY(-3px); /* 마우스 올리면 살짝 위로 */
}

.prob-label {
    font-weight: bold;
    font-size: 1.1rem;
    color: #333;
}

.prob-bar-bg {
    background-color: #E0E0E0; /* Light Gray */
    border-radius: 5px;
    width: 100%;
    height: 22px;
    overflow: hidden; /* 둥근 모서리 적용 */
}

.prob-bar-fg {
    background-color: #4CAF50; /* Green */
    height: 100%;
    border-radius: 5px 0 0 5px; /* 왼쪽만 둥글게 */
    text-align: right;
    padding-right: 8px;
    color: white;
    font-weight: bold;
    line-height: 22px; /* 텍스트 세로 중앙 정렬 */
    transition: width 0.5s ease-in-out; /* 너비 변경 애니메이션 */
}

/* 가장 높은 확률의 바 */
.prob-bar-fg.highlight {
    background-color: #FF6F00; /* Orange */
}
</style>
""", unsafe_allow_html=True)


# --- 3. 모델 로드 ---

# Google Drive 파일 ID
file_id = '19dS6rAzHlGekODz1l2F020D9XMlhNDYS'
model_path = 'model.pkl'

# st.cache_resource: 모델과 같이 큰 리소스를 캐시합니다.
@st.cache_resource
def load_model_from_drive(file_id, output_path):
    # 파일이 이미 존재하지 않으면 다운로드
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output_path)
    return learner

# st.spinner: 모델 로딩 중에 스피너(빙글빙글 아이콘)를 보여줍니다.
with st.spinner("🤖 AI 모델을 불러오는 중입니다. 잠시만 기다려주세요..."):
    learner = load_model_from_drive(file_id, model_path)

st.success("✅ 모델 로드가 완료되었습니다!")

# 모델의 분류 라벨
labels = learner.dls.vocab
st.title(f"이미지 분류기 (Fastai)")
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# --- 4. 파일 업로드 및 예측 ---
uploaded_file = st.file_uploader("분류할 이미지를 업로드하세요 (jpg, png, jpeg 등)", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    # 업로드된 이미지 보여주기
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # Fastai에서 예측을 위해 이미지를 처리
    # (주의: PILImage.create는 fastai 구버전일 수 있습니다. 최신 버전에 맞춰 바이트로 열기)
    try:
        img_bytes = uploaded_file.getvalue()
        img = PILImage.create(img_bytes)
    except Exception as e:
        st.error(f"이미지 처리 중 오류 발생: {e}")
        st.stop()

    # 예측 수행
    with st.spinner("🧠 이미지를 분석 중입니다..."):
        prediction, pred_idx, probs = learner.predict(img)

    # --- 5. 결과 출력 ---

    # 예측된 클래스 출력 (스타일 적용)
    st.markdown(f"""
    <div class="prediction-box">
        <span style="font-size: 1.2rem; color: #555;">예측 결과:</span>
        <h2>{prediction}</h2>
    </div>
    """, unsafe_allow_html=True)


    # 클래스별 확률을 HTML과 CSS로 시각화
    st.markdown("<h3>상세 예측 확률:</h3>", unsafe_allow_html=True)

    # 확률을 내림차순으로 정렬 (시각적으로 보기 좋게)
    prob_list = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)

    for label, prob in prob_list:
        # 가장 높은 확률(예측된 클래스)인지 확인
        highlight_class = "highlight" if label == prediction else ""

        prob_percent = prob * 100

        st.markdown(f"""
        <div class="prob-card">
            <span class="prob-label">{label}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fg {highlight_class}" style="width: {prob_percent}%;">
                    {prob_percent:.2f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("이미지를 업로드하면 AI가 분석을 시작합니다.")

