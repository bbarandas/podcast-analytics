import streamlit as st

st.set_page_config(
    page_title="Podcast Analytics — Futebol",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Podcast Analytics")
st.markdown("""
Bem-vindo ao toolkit de analytics do podcast.

**Navegue pelas páginas no menu lateral:**
- **Posicionamento** — mapas de calor, passes e pressão (StatsBomb Open Data)
- **Vídeo** — detecção de jogadores em lances específicos
- **Biomecânica** — análise de pose e ângulos articulares
""")
