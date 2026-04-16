import streamlit as st
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent

from modules.video.downloader import download, extract_frames
from modules.video.detector import detect_players, annotate_frames

st.set_page_config(page_title="Vídeo", layout="wide")
st.title("🎥 Análise de Vídeo")

st.sidebar.header("Fonte de vídeo")
source = st.sidebar.radio("Origem", ["Upload de arquivo", "URL do YouTube"])

video_path = None

if source == "Upload de arquivo":
    uploaded = st.sidebar.file_uploader("Envie um MP4", type=["mp4"])
    if uploaded:
        tmp = Path(tempfile.mkdtemp()) / uploaded.name
        tmp.write_bytes(uploaded.read())
        video_path = tmp
else:
    url = st.sidebar.text_input("URL do YouTube")
    if st.sidebar.button("Baixar vídeo") and url:
        with st.spinner("Baixando..."):
            dl_dir = PROJECT_ROOT / "data" / "video_downloads"
            video_path = download(url, dl_dir)
            st.session_state["video_path"] = str(video_path)
            st.success(f"Baixado: {video_path.name}")

if "video_path" in st.session_state and video_path is None:
    video_path = Path(st.session_state["video_path"])

if video_path is None:
    st.info("Selecione um vídeo para continuar.")
    st.stop()

st.video(str(video_path))

st.subheader("Extrair frames do lance")
col1, col2 = st.columns(2)
start_sec = col1.number_input("Início (segundos)", min_value=0.0, value=0.0, step=0.5)
end_sec = col2.number_input("Fim (segundos)", min_value=0.0, value=5.0, step=0.5)

if st.button("Extrair frames e detectar jogadores"):
    with st.spinner("Extraindo frames..."):
        frames_dir = PROJECT_ROOT / "data" / "frames" / video_path.stem
        frames = extract_frames(video_path, start_sec, end_sec, frames_dir)
        st.success(f"{len(frames)} frames extraídos.")

    with st.spinner("Detectando jogadores (YOLO)..."):
        detections = detect_players(frames_dir)
        annotated_dir = PROJECT_ROOT / "data" / "annotated" / video_path.stem
        annotated = annotate_frames(detections, annotated_dir)
        st.session_state["frames"] = [str(p) for p in frames]
        st.session_state["annotated"] = [str(p) for p in annotated]
        st.success("Detecção concluída.")

if "annotated" in st.session_state:
    st.subheader("Frames anotados")
    annotated_paths = [Path(p) for p in st.session_state["annotated"]]
    # Show a sample of frames (every 5th to avoid overloading)
    sample = annotated_paths[::5]
    cols = st.columns(min(4, len(sample)))
    for i, fp in enumerate(sample[:4]):
        cols[i].image(str(fp), use_container_width=True)

    st.info("Vá para a aba **Biomecânica** para analisar os ângulos articulares.")
