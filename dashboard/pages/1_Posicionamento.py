import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.data_sources.statsbomb import list_competitions, list_matches, get_events
from modules.positioning.pitch_maps import heatmap, pass_map, pressure_map, avg_positions
from visualizations.exporter import export_png

st.set_page_config(page_title="Posicionamento", layout="wide")
st.title("📍 Posicionamento")

# --- Sidebar: data selection ---
st.sidebar.header("Selecionar Partida")

@st.cache_data
def load_competitions():
    return list_competitions()

@st.cache_data
def load_matches(competition_id, season_id):
    return list_matches(competition_id=competition_id, season_id=season_id)

@st.cache_data
def load_events(match_id):
    return get_events(match_id=match_id)

competitions = load_competitions()
comp_options = competitions.apply(
    lambda r: f"{r['competition_name']} — {r['season_name']}", axis=1
).tolist()
selected_comp_label = st.sidebar.selectbox("Competição / Temporada", comp_options)
selected_comp_idx = comp_options.index(selected_comp_label)
selected_comp = competitions.iloc[selected_comp_idx]

matches = load_matches(
    competition_id=int(selected_comp["competition_id"]),
    season_id=int(selected_comp["season_id"]),
)
match_options = matches.apply(
    lambda r: f"{r['home_team']} vs {r['away_team']} ({r['match_date']})", axis=1
).tolist()
selected_match_label = st.sidebar.selectbox("Partida", match_options)
selected_match_idx = match_options.index(selected_match_label)
selected_match = matches.iloc[selected_match_idx]

if st.sidebar.button("Carregar dados"):
    st.session_state["events"] = load_events(match_id=int(selected_match["match_id"]))
    st.success("Dados carregados!")

if "events" not in st.session_state:
    st.info("Selecione uma partida e clique em 'Carregar dados'.")
    st.stop()

events: pd.DataFrame = st.session_state["events"]

# --- Visualization controls ---
viz_type = st.selectbox(
    "Visualização",
    ["Mapa de Calor", "Mapa de Passes", "Mapa de Pressão", "Posição Média"],
)

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Filtros")
    teams = sorted(events["team"].dropna().unique().tolist())
    selected_team = st.selectbox("Time", teams)
    players = sorted(
        events[events["team"] == selected_team]["player"].dropna().unique().tolist()
    )
    selected_player = st.selectbox("Jogador", players) if players else None

with col1:
    fig = None
    if viz_type == "Mapa de Calor" and selected_player:
        fig = heatmap(events, player_name=selected_player)
    elif viz_type == "Mapa de Passes" and selected_player:
        fig = pass_map(events, player_name=selected_player)
    elif viz_type == "Mapa de Pressão":
        fig = pressure_map(events, team_name=selected_team)
    elif viz_type == "Posição Média":
        fig = avg_positions(events, team_name=selected_team)

    if fig:
        st.pyplot(fig)
        safe_name = (selected_player or selected_team).replace(" ", "_").lower()
        export_filename = f"{viz_type.replace(' ', '_').lower()}_{safe_name}.png"
        if st.button("Exportar PNG"):
            path = export_png(fig, export_filename)
            st.success(f"Salvo em {path}")
