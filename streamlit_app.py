# streamlit_app.py
import os
from io import BytesIO
import re
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 0) 페이지/스타일 설정
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기 (스냅샷+정보패널)", page_icon="🤖", layout="wide")

st.markdown("""
<style>
h1 { color: #1E88E5; text-align: center; font-weight: 800; letter-spacing: -0.5px; }
.stFileUploader, .stCameraInput {
  border: 2px dashed #1E88E5; border-radius: 12px; padding: 16px; background-color: #f5fafe;
}
hr { margin: 1rem 0; }
.prediction-box {
  background-color: #E3F2FD; border: 2px solid #1E88E5; border-radius: 12px;
  padding: 22px; text-align: center; margin: 16px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.06);
}
.prediction-box h2 { color: #0D47A1; margin: 0; font-size: 2.0rem; }
.prob-card {
  background-color: #FFFFFF; border-radius: 10px; padding: 12px 14px; margin: 10px 0;
  box-shadow: 0 2px 6px rgba(0,0,0,0.06); transition: transform 0.2s ease;
}
.prob-card:hover { transform: translateY(-2px); }
.prob-label { font-weight: 700; font-size: 1.0rem; color: #333; display: inline-block; margin-bottom: 6px; }
.prob-bar-bg { background-color: #ECEFF1; border-radius: 6px; width: 100%; height: 22px; overflow: hidden; }
.prob-bar-fg {
  background-color: #4CAF50; height: 100%; border-radius: 6px; text-align: right;
  padding-right: 8px; color: white; font-weight: 700; line-height: 22px; transition: width 0.5s ease-in-out;
}
.prob-bar-fg.highlight { background-color: #FF6F00; }
.info-grid {
  display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; align-items: start;
}
.card {
  border: 1px solid #e3e6ea; border-radius: 12px; padding: 14px; background: #fff;
  box-shadow: 0 2px 6px rgba(0,0,0,.05);
}
.card h4 { margin: 0 0 10px; font-size: 1.05rem; color: #0D47A1; }
.badge {
  display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; color: #0D47A1;
  background: #E3F2FD; border: 1px solid #BBDEFB; margin-left: 8px;
}
.thumb { width: 100%; height: auto; border-radius: 10px; display:block; }
.thumb-wrap { position: relative; }
.play {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  width: 60px; height: 60px; border-radius: 50%; background: rgba(0,0,0,.55);
  display:flex; align-items:center; justify-content:center;
}
.play:after {
  content: ''; border-style: solid; border-width: 12px 0 12px 20px; border-color: transparent transparent transparent white; margin-left: 3px;
}
.kv { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:6px; }
.kv .k { font-weight:700; color:#455A64; }
.kv .v { color:#263238; }
.helper { color:#607D8B; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨 기반 정보 패널")

# ======================
# 1) 세션 상태 초기화
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "info_config" not in st.session_state:
    # { label: {"texts":[], "images":[], "videos":[]} }
    st.session_state.info_config = {}
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 2) 모델 로드 (Google Drive)
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "19dS6rAzHlGekODz1l2F020D9XMlhNDYS")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    learner = load_learner(output_path, cpu=True)
    return learner

with st.spinner("🤖 AI 모델을 불러오는 중입니다..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 유틸: 이미지 로딩, YouTube 파싱
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    # 다양한 YouTube 링크 패턴 지원
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|\/|$)",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

# ======================
# 3) 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])

new_bytes = None
with tab_cam:
    st.write("카메라 권한을 허용한 뒤, 스냅샷을 촬영하세요.")
    camera_photo = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if camera_photo is not None:
        new_bytes = camera_photo.getvalue()

with tab_file:
    uploaded_file = st.file_uploader(
        "이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
        type=["jpg", "png", "jpeg", "webp", "tiff"]
    )
    if uploaded_file is not None:
        new_bytes = uploaded_file.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 4) 사이드바: 라벨별 정보 설정
# ======================
with st.sidebar:
    st.header("설정")
    st.caption("오른쪽 정보 패널에 표시할 라벨별 콘텐츠(텍스트/이미지/동영상)를 등록하세요.")
    # 동적 라벨 선택
    label_sel = st.selectbox("라벨 선택", options=labels, index=0 if labels else 0)
    st.markdown('<div class="helper">각 항목별 최대 3개까지 등록됩니다.</div>', unsafe_allow_html=True)

    def get_cfg(lbl: str):
        return st.session_state.info_config.get(lbl, {"texts": [], "images": [], "videos": []})

    cfg = get_cfg(label_sel)

    with st.form("content_form", clear_on_submit=False):
        st.subheader(f"콘텐츠 입력 — {label_sel}")
        t1 = st.text_input("텍스트 #1", value=cfg["texts"][0] if len(cfg["texts"]) > 0 else "")
        t2 = st.text_input("텍스트 #2", value=cfg["texts"][1] if len(cfg["texts"]) > 1 else "")
        t3 = st.text_input("텍스트 #3", value=cfg["texts"][2] if len(cfg["texts"]) > 2 else "")

        i1 = st.text_input("이미지 URL #1", value=cfg["images"][0] if len(cfg["images"]) > 0 else "")
        i2 = st.text_input("이미지 URL #2", value=cfg["images"][1] if len(cfg["images"]) > 1 else "")
        i3 = st.text_input("이미지 URL #3", value=cfg["images"][2] if len(cfg["images"]) > 2 else "")

        v1 = st.text_input("YouTube 링크 #1", value=cfg["videos"][0] if len(cfg["videos"]) > 0 else "")
        v2 = st.text_input("YouTube 링크 #2", value=cfg["videos"][1] if len(cfg["videos"]) > 1 else "")
        v3 = st.text_input("YouTube 링크 #3", value=cfg["videos"][2] if len(cfg["videos"]) > 2 else "")

        submitted = st.form_submit_button("저장/업데이트")
        if submitted:
            new_cfg = {
                "texts": [x for x in [t1, t2, t3] if x.strip() != ""],
                "images": [x for x in [i1, i2, i3] if x.strip() != ""],
                "videos": [x for x in [v1, v2, v3] if x.strip() != ""],
            }
            st.session_state.info_config[label_sel] = new_cfg
            st.success(f"라벨 `{label_sel}` 콘텐츠가 저장되었습니다.")

    st.divider()
    st.caption("모델 파일은 최초 1회 다운로드 후 캐시됩니다.")
    st.write(f"**모델 파일**: `{MODEL_PATH}`")
    st.write(f"**Drive File ID**: `{FILE_ID}`")

# ======================
# 5) 예측 파이프라인 + 레이아웃 (왼쪽: 확률, 오른쪽: 정보패널)
# ======================
if st.session_state.img_bytes:
    # 상단에 입력 이미지와 예측 결과 박스
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    try:
        pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    except Exception as e:
        st.exception(e)
        st.stop()

    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    # fastai 입력
    try:
        fa_img = PILImage.create(np.array(pil_img))
    except Exception as e:
        st.exception(e)
        st.stop()

    with st.spinner("🧠 이미지를 분석 중입니다..."):
        try:
            prediction, pred_idx, probs = learner.predict(fa_img)
            st.session_state.last_prediction = str(prediction)
        except Exception as e:
            st.exception(e)
            st.stop()

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size: 1.0rem; color: #555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 다른 라벨을 선택하여 정보 보기 가능</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 메인 2컬럼: L=확률 막대, R=정보패널
    col_left, col_right = st.columns([1, 1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with col_left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1],
            reverse=True
        )
        for label, prob in prob_list:
            highlight_class = "highlight" if label == st.session_state.last_prediction else ""
            prob_percent = prob * 100.0
            st.markdown(
                f"""
                <div class="prob-card">
                    <div class="kv"><div class="k">{label}</div><div class="v">{prob_percent:.2f}%</div></div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fg {highlight_class}" style="width: {prob_percent:.4f}%;">
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 사용자가 다른 라벨로 전환 가능)
    with col_right:
        st.subheader("라벨 기반 정보 패널")

        # 기본 선택은 예측 라벨, 필요시 변경 가능
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("정보를 볼 라벨 선택", options=labels, index=default_idx)

        cfg = st.session_state.info_config.get(info_label, {"texts": [], "images": [], "videos": []})

        # 카드 그리드 표시: 텍스트/이미지/영상 각각 최대 3개
        st.markdown(
            f'<div class="kv"><div class="k">현재 라벨</div><div class="v">{info_label}</div>'
            f'<span class="badge">texts: {len(cfg["texts"])}</span>'
            f'<span class="badge">images: {len(cfg["images"])}</span>'
            f'<span class="badge">videos: {len(cfg["videos"])}</span></div>',
            unsafe_allow_html=True
        )

        # 텍스트 카드
        if cfg["texts"]:
            st.markdown('<div class="info-grid">', unsafe_allow_html=True)
            for txt in cfg["texts"][:3]:
                st.markdown(f"""
                <div class="card" style="grid-column: span 12;">
                    <h4>텍스트</h4>
                    <div>{txt}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 이미지 카드 (3열 그리드)
        if cfg["images"]:
            st.markdown('<div class="info-grid">', unsafe_allow_html=True)
            for url in cfg["images"][:3]:
                st.markdown(f"""
                <div class="card" style="grid-column: span 4;">
                    <h4>이미지</h4>
                    <img src="{url}" alt="image" class="thumb" />
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 동영상 카드(YouTube 썸네일)
        if cfg["videos"]:
            st.markdown('<div class="info-grid">', unsafe_allow_html=True)
            for url in cfg["videos"][:3]:
                thumb = yt_thumb(url)
                if thumb:
                    st.markdown(f"""
                    <div class="card" style="grid-column: span 6;">
                        <h4>동영상</h4>
                        <a href="{url}" target="_blank" rel="noopener noreferrer" class="thumb-wrap">
                            <img src="{thumb}" alt="video" class="thumb" />
                            <div class="play"></div>
                        </a>
                        <div class="helper">{url}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="card" style="grid-column: span 6;">
                        <h4>동영상</h4>
                        <div class="helper">YouTube 링크가 아닌 경우 썸네일을 생성하지 않습니다.</div>
                        <a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>
                    </div>
                    """, unsafe_allow_html=True)

else:
    st.info("카메라에서 스냅샷을 촬영하거나, 파일을 업로드하면 AI가 분석을 시작합니다.")
