# Podcast Analytics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python toolkit + Streamlit dashboard for football analytics, supporting heatmaps/pass maps from StatsBomb Open Data (Phase 1), and video-based player detection and biomechanical pose analysis (Phase 2).

**Architecture:** Modular Python packages under `modules/` (data_sources, positioning, video, biomechanics) with a shared `visualizations/` layer for themed exports. A Streamlit multi-page app in `dashboard/` ties everything together with interactive controls and PNG export buttons.

**Tech Stack:** Python 3.11+, statsbombpy, mplsoccer, matplotlib, plotly, streamlit, yt-dlp, opencv-python, ultralytics (YOLOv8), mediapipe, pytest

---

## File Map

```
podcast-analytics/
├── modules/
│   ├── __init__.py
│   ├── data_sources/
│   │   ├── __init__.py
│   │   └── statsbomb.py          # list_competitions, list_matches, get_events, get_360
│   ├── positioning/
│   │   ├── __init__.py
│   │   └── pitch_maps.py         # heatmap, pass_map, pressure_map, avg_positions
│   ├── video/
│   │   ├── __init__.py
│   │   ├── downloader.py         # download(url), extract_frames(video, start, end)
│   │   └── detector.py           # detect_players(frames_dir), annotate_video(...)
│   └── biomechanics/
│       ├── __init__.py
│       └── pose.py               # estimate_pose, calculate_angles, overlay_skeleton
├── visualizations/
│   ├── __init__.py
│   ├── theme.py                  # PITCH_BG, LINE_COLOR, ACCENT colors
│   └── exporter.py               # export_png(fig, filename) → Path
├── dashboard/
│   ├── app.py                    # Streamlit entry point + sidebar
│   └── pages/
│       ├── 1_Posicionamento.py   # Heatmap/pass/pressure/avg position page
│       ├── 2_Video.py            # Upload + frame extraction + YOLO page
│       └── 3_Biomecanica.py      # Pose estimation + angle charts page
├── tests/
│   ├── test_statsbomb.py
│   ├── test_positioning.py
│   ├── test_exporter.py
│   ├── test_video.py
│   └── test_biomechanics.py
├── data/                         # gitignored — cached StatsBomb data
├── exports/                      # gitignored — generated PNGs
├── requirements.txt
└── .gitignore
```

---

## PHASE 1 — Positioning Pipeline + Dashboard

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `modules/__init__.py` (empty)
- Create: `modules/data_sources/__init__.py` (empty)
- Create: `modules/positioning/__init__.py` (empty)
- Create: `modules/video/__init__.py` (empty)
- Create: `modules/biomechanics/__init__.py` (empty)
- Create: `visualizations/__init__.py` (empty)
- Create: `dashboard/pages/.gitkeep` (empty)

- [ ] **Step 1: Create requirements.txt**

```text
statsbombpy
mplsoccer
matplotlib
plotly
streamlit
pandas
numpy
Pillow
yt-dlp
opencv-python
mediapipe
ultralytics
pytest
```

- [ ] **Step 2: Create .gitignore**

```text
data/
exports/
__pycache__/
*.pyc
*.mp4
*.avi
.env
.DS_Store
```

- [ ] **Step 3: Create all empty __init__.py files and dirs**

```bash
mkdir -p modules/data_sources modules/positioning modules/video modules/biomechanics
mkdir -p visualizations dashboard/pages tests data exports
touch modules/__init__.py
touch modules/data_sources/__init__.py
touch modules/positioning/__init__.py
touch modules/video/__init__.py
touch modules/biomechanics/__init__.py
touch visualizations/__init__.py
touch dashboard/pages/.gitkeep
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore modules/ visualizations/ dashboard/ tests/ data/ exports/
git commit -m "chore: scaffold project structure"
```

---

### Task 2: StatsBomb Connector

**Files:**
- Create: `modules/data_sources/statsbomb.py`
- Create: `tests/test_statsbomb.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_statsbomb.py`:

```python
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from modules.data_sources.statsbomb import list_competitions, list_matches, get_events, get_360


@patch("modules.data_sources.statsbomb.sb.competitions")
def test_list_competitions_returns_dataframe(mock_comps):
    mock_comps.return_value = pd.DataFrame([{"competition_id": 11, "season_id": 1, "competition_name": "La Liga"}])
    result = list_competitions()
    assert isinstance(result, pd.DataFrame)
    assert "competition_id" in result.columns


@patch("modules.data_sources.statsbomb.sb.matches")
def test_list_matches_passes_ids(mock_matches):
    mock_matches.return_value = pd.DataFrame([{"match_id": 3788741}])
    result = list_matches(competition_id=11, season_id=1)
    mock_matches.assert_called_once_with(competition_id=11, season_id=1)
    assert isinstance(result, pd.DataFrame)


@patch("modules.data_sources.statsbomb.sb.events")
def test_get_events_returns_dataframe(mock_events):
    mock_events.return_value = pd.DataFrame([{"type": "Pass", "player": "Messi", "team": "Barcelona"}])
    result = get_events(match_id=3788741)
    mock_events.assert_called_once_with(match_id=3788741)
    assert isinstance(result, pd.DataFrame)


@patch("modules.data_sources.statsbomb.sb.frames")
def test_get_360_returns_dataframe(mock_frames):
    mock_frames.return_value = pd.DataFrame([{"id": "abc", "freeze_frame": []}])
    result = get_360(match_id=3788741)
    mock_frames.assert_called_once_with(match_id=3788741)
    assert isinstance(result, pd.DataFrame)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/projects/podcast-analytics
pytest tests/test_statsbomb.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `statsbomb.py` doesn't exist yet.

- [ ] **Step 3: Create modules/data_sources/statsbomb.py**

```python
from statsbombpy import sb
import pandas as pd


def list_competitions() -> pd.DataFrame:
    """Return all available StatsBomb Open Data competitions."""
    return sb.competitions()


def list_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return all matches for a competition/season."""
    return sb.matches(competition_id=competition_id, season_id=season_id)


def get_events(match_id: int) -> pd.DataFrame:
    """Return all events for a match as a flat DataFrame."""
    return sb.events(match_id=match_id)


def get_360(match_id: int) -> pd.DataFrame:
    """Return 360 freeze-frame positional data for a match."""
    return sb.frames(match_id=match_id)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_statsbomb.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/data_sources/statsbomb.py tests/test_statsbomb.py
git commit -m "feat: add StatsBomb Open Data connector"
```

---

### Task 3: Visualization Theme

**Files:**
- Create: `visualizations/theme.py`

No isolated unit tests needed — constants only. Tested implicitly by positioning tests.

- [ ] **Step 1: Create visualizations/theme.py**

```python
# Podcast analytics visual identity
PITCH_BG = "#1a1a2e"       # dark navy
LINE_COLOR = "#e0e0e0"     # light gray
ACCENT_GREEN = "#00ff88"   # successful pass / positive
ACCENT_RED = "#ff4444"     # failed pass / negative
ACCENT_BLUE = "#00bfff"    # player markers
TEXT_COLOR = "#ffffff"
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 7
```

- [ ] **Step 2: Commit**

```bash
git add visualizations/theme.py
git commit -m "feat: add podcast visual theme constants"
```

---

### Task 4: Positioning Module

**Files:**
- Create: `modules/positioning/pitch_maps.py`
- Create: `tests/test_positioning.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_positioning.py`:

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering for tests
from matplotlib.figure import Figure
from modules.positioning.pitch_maps import heatmap, pass_map, pressure_map, avg_positions


def _make_events() -> pd.DataFrame:
    """Minimal events DataFrame mimicking StatsBomb structure."""
    return pd.DataFrame([
        {"type": "Pass", "player": "Messi", "team": "Barcelona",
         "location": [30.0, 40.0], "pass_end_location": [60.0, 50.0], "pass_outcome": float("nan")},
        {"type": "Pass", "player": "Messi", "team": "Barcelona",
         "location": [50.0, 30.0], "pass_end_location": [80.0, 20.0], "pass_outcome": "Incomplete"},
        {"type": "Pressure", "player": "Messi", "team": "Barcelona",
         "location": [70.0, 40.0], "pass_end_location": None, "pass_outcome": None},
        {"type": "Carry", "player": "Xavi", "team": "Barcelona",
         "location": [45.0, 35.0], "pass_end_location": None, "pass_outcome": None},
    ])


def test_heatmap_returns_figure():
    events = _make_events()
    fig = heatmap(events, player_name="Messi")
    assert isinstance(fig, Figure)


def test_pass_map_returns_figure():
    events = _make_events()
    fig = pass_map(events, player_name="Messi")
    assert isinstance(fig, Figure)


def test_pressure_map_returns_figure():
    events = _make_events()
    fig = pressure_map(events, team_name="Barcelona")
    assert isinstance(fig, Figure)


def test_avg_positions_returns_figure():
    events = _make_events()
    fig = avg_positions(events, team_name="Barcelona")
    assert isinstance(fig, Figure)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_positioning.py -v
```

Expected: `ImportError` — `pitch_maps.py` doesn't exist yet.

- [ ] **Step 3: Create modules/positioning/pitch_maps.py**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mplsoccer import Pitch
from visualizations.theme import (
    PITCH_BG, LINE_COLOR, ACCENT_GREEN, ACCENT_RED,
    ACCENT_BLUE, TEXT_COLOR, FONT_SIZE_TITLE, FONT_SIZE_LABEL
)


def _base_pitch() -> tuple:
    """Return (Pitch, fig, ax) with podcast theme applied."""
    pitch = Pitch(pitch_color="grass", line_color=LINE_COLOR, line_zorder=2)
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.set_facecolor(PITCH_BG)
    return pitch, fig, ax


def heatmap(events: pd.DataFrame, player_name: str) -> Figure:
    """Player heatmap based on all event locations."""
    player_events = events[events["player"] == player_name].dropna(subset=["location"])
    locs = pd.DataFrame(player_events["location"].tolist(), columns=["x", "y"])

    pitch, fig, ax = _base_pitch()
    if len(locs) >= 2:
        pitch.kdeplot(locs.x, locs.y, ax=ax, cmap="plasma", fill=True, levels=100, zorder=3)
    ax.set_title(f"Mapa de Calor — {player_name}", fontsize=FONT_SIZE_TITLE, color=TEXT_COLOR)
    return fig


def pass_map(events: pd.DataFrame, player_name: str) -> Figure:
    """Pass map showing successful (green) and failed (red) passes."""
    passes = events[
        (events["player"] == player_name) & (events["type"] == "Pass")
    ].dropna(subset=["location", "pass_end_location"])

    pitch, fig, ax = _base_pitch()
    for _, row in passes.iterrows():
        x, y = row["location"]
        ex, ey = row["pass_end_location"]
        color = ACCENT_GREEN if pd.isna(row.get("pass_outcome")) else ACCENT_RED
        pitch.arrows(x, y, ex, ey, ax=ax, color=color, width=1.5, headwidth=5, zorder=2)

    ax.set_title(f"Mapa de Passes — {player_name}", fontsize=FONT_SIZE_TITLE, color=TEXT_COLOR)
    return fig


def pressure_map(events: pd.DataFrame, team_name: str) -> Figure:
    """Defensive pressure heatmap for a team."""
    pressures = events[
        (events["team"] == team_name) & (events["type"] == "Pressure")
    ].dropna(subset=["location"])
    locs = pd.DataFrame(pressures["location"].tolist(), columns=["x", "y"])

    pitch, fig, ax = _base_pitch()
    if len(locs) >= 2:
        pitch.kdeplot(locs.x, locs.y, ax=ax, cmap="Reds", fill=True, levels=100, zorder=3, alpha=0.8)
    ax.set_title(f"Mapa de Pressão — {team_name}", fontsize=FONT_SIZE_TITLE, color=TEXT_COLOR)
    return fig


def avg_positions(events: pd.DataFrame, team_name: str) -> Figure:
    """Average position of each player on the team."""
    team_events = events[(events["team"] == team_name)].dropna(subset=["location", "player"])
    locs = team_events.copy()
    locs[["x", "y"]] = pd.DataFrame(locs["location"].tolist(), index=locs.index)
    avg = locs.groupby("player")[["x", "y"]].mean()

    pitch, fig, ax = _base_pitch()
    pitch.scatter(avg.x, avg.y, ax=ax, s=200, color=ACCENT_BLUE, zorder=5, edgecolors="white", linewidths=1.5)
    for player, row in avg.iterrows():
        ax.annotate(
            player.split()[-1], (row.x, row.y),
            fontsize=FONT_SIZE_LABEL, color=TEXT_COLOR,
            ha="center", va="bottom", xytext=(0, 8), textcoords="offset points"
        )
    ax.set_title(f"Posição Média — {team_name}", fontsize=FONT_SIZE_TITLE, color=TEXT_COLOR)
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_positioning.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/positioning/pitch_maps.py tests/test_positioning.py
git commit -m "feat: add positioning module (heatmap, pass map, pressure map, avg positions)"
```

---

### Task 5: PNG Exporter

**Files:**
- Create: `visualizations/exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_exporter.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pytest
from visualizations.exporter import export_png


def _dummy_fig():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def test_export_png_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("visualizations.exporter.EXPORTS_DIR", tmp_path)
    fig = _dummy_fig()
    result = export_png(fig, "test_output.png")
    assert result.exists()
    assert result.suffix == ".png"


def test_export_png_returns_path(tmp_path, monkeypatch):
    monkeypatch.setattr("visualizations.exporter.EXPORTS_DIR", tmp_path)
    fig = _dummy_fig()
    result = export_png(fig, "test_output2.png")
    assert isinstance(result, Path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_exporter.py -v
```

Expected: `ImportError` — `exporter.py` doesn't exist yet.

- [ ] **Step 3: Create visualizations/exporter.py**

```python
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

EXPORTS_DIR = Path(__file__).parent.parent / "exports"


def export_png(fig: Figure, filename: str) -> Path:
    """Save a matplotlib Figure as a PNG to the exports directory.

    Args:
        fig: The matplotlib Figure to export.
        filename: Output filename (e.g. 'heatmap_messi.png').

    Returns:
        Path to the saved file.
    """
    EXPORTS_DIR.mkdir(exist_ok=True)
    output_path = EXPORTS_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_exporter.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add visualizations/exporter.py tests/test_exporter.py
git commit -m "feat: add PNG exporter with exports/ output directory"
```

---

### Task 6: Streamlit Dashboard — Posicionamento Page

**Files:**
- Create: `dashboard/app.py`
- Create: `dashboard/pages/1_Posicionamento.py`

No unit tests for Streamlit pages — test by running the app manually.

- [ ] **Step 1: Create dashboard/app.py**

```python
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
```

- [ ] **Step 2: Create dashboard/pages/1_Posicionamento.py**

```python
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
```

- [ ] **Step 3: Run the dashboard and test manually**

```bash
cd ~/projects/podcast-analytics
streamlit run dashboard/app.py
```

Abra o browser em `http://localhost:8501`. Navegue até **Posicionamento**. Selecione uma competição (ex: "Champions League — 2018/2019"), uma partida, clique em "Carregar dados". Selecione "Mapa de Calor" e um jogador. Verifique que o mapa aparece. Clique "Exportar PNG" e confirme que o arquivo aparece em `exports/`.

- [ ] **Step 4: Commit**

```bash
git add dashboard/app.py dashboard/pages/1_Posicionamento.py
git commit -m "feat: add Streamlit Posicionamento page with heatmap/pass/pressure/avg-position"
```

> **✅ PHASE 1 COMPLETE** — Dashboard de posicionamento funcional. Você já pode usar para preparar episódios sobre futebol europeu.

---

## PHASE 2 — Video + Biomechanics Pipeline

---

### Task 7: Video Downloader + Frame Extractor

**Files:**
- Create: `modules/video/downloader.py`
- Create: `tests/test_video.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_video.py`:

```python
import cv2
import numpy as np
import pytest
from pathlib import Path
from modules.video.downloader import extract_frames


def _create_test_video(path: Path, num_frames: int = 30, fps: int = 10):
    """Create a minimal synthetic MP4 video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for i in range(num_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 8  # vary color so frames differ
        out.write(frame)
    out.release()


def test_extract_frames_creates_jpegs(tmp_path):
    video_path = tmp_path / "test.mp4"
    _create_test_video(video_path, num_frames=30, fps=10)
    output_dir = tmp_path / "frames"

    frames = extract_frames(video_path, start_sec=0.0, end_sec=1.0, output_dir=output_dir)

    assert len(frames) > 0
    assert all(p.suffix == ".jpg" for p in frames)
    assert all(p.exists() for p in frames)


def test_extract_frames_respects_time_range(tmp_path):
    video_path = tmp_path / "test.mp4"
    _create_test_video(video_path, num_frames=30, fps=10)
    output_dir = tmp_path / "frames"

    frames_full = extract_frames(video_path, start_sec=0.0, end_sec=3.0, output_dir=output_dir)
    frames_half = extract_frames(video_path, start_sec=0.0, end_sec=1.0, output_dir=tmp_path / "frames2")

    assert len(frames_full) > len(frames_half)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_video.py -v
```

Expected: `ImportError` — `downloader.py` doesn't exist yet.

- [ ] **Step 3: Create modules/video/downloader.py**

```python
import subprocess
from pathlib import Path
from typing import List
import cv2


def download(url: str, output_dir: Path) -> Path:
    """Download a YouTube video as MP4 to output_dir.

    Args:
        url: YouTube URL.
        output_dir: Directory to save the downloaded file.

    Returns:
        Path to the downloaded MP4 file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s.%(ext)s")
    subprocess.run(
        ["yt-dlp", "-o", output_template, "--format", "mp4", url],
        check=True,
    )
    mp4_files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return mp4_files[-1]


def extract_frames(
    video_path: Path, start_sec: float, end_sec: float, output_dir: Path
) -> List[Path]:
    """Extract frames from a video between start_sec and end_sec.

    Args:
        video_path: Path to the MP4 file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        output_dir: Directory to save extracted JPEG frames.

    Returns:
        List of paths to extracted frame JPEGs, sorted by frame number.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_paths: List[Path] = []
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        frame_idx += 1

    cap.release()
    return frame_paths
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_video.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/video/downloader.py tests/test_video.py
git commit -m "feat: add video downloader and frame extractor"
```

---

### Task 8: YOLO Player Detector

**Files:**
- Create: `modules/video/detector.py`

No isolated unit test for YOLO — requires GPU/model download and real images. Tested via the dashboard in Task 10.

- [ ] **Step 1: Create modules/video/detector.py**

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import cv2
from ultralytics import YOLO


@dataclass
class Detection:
    frame_path: Path
    # Each box: [x1, y1, x2, y2, confidence, class_id]
    boxes: List[List[float]] = field(default_factory=list)


def detect_players(frames_dir: Path, model_name: str = "yolov8n.pt") -> List[Detection]:
    """Run YOLO person detection on all JPEGs in frames_dir.

    Args:
        frames_dir: Directory containing frame JPEGs.
        model_name: YOLOv8 model variant (yolov8n.pt = nano, fastest).

    Returns:
        List of Detection objects, one per frame.
    """
    model = YOLO(model_name)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    detections: List[Detection] = []

    for frame_path in frame_paths:
        results = model(str(frame_path), classes=[0], verbose=False)  # class 0 = person
        boxes = results[0].boxes.data.tolist() if results[0].boxes is not None else []
        detections.append(Detection(frame_path=frame_path, boxes=boxes))

    return detections


def annotate_frames(detections: List[Detection], output_dir: Path) -> List[Path]:
    """Draw bounding boxes on frames and save to output_dir.

    Args:
        detections: List of Detection objects.
        output_dir: Directory to save annotated JPEGs.

    Returns:
        List of paths to annotated frame JPEGs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []

    for det in detections:
        frame = cv2.imread(str(det.frame_path))
        for i, box in enumerate(det.boxes):
            x1, y1, x2, y2, conf, _ = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame, f"#{i} {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        out_path = output_dir / det.frame_path.name
        cv2.imwrite(str(out_path), frame)
        output_paths.append(out_path)

    return output_paths
```

- [ ] **Step 2: Commit**

```bash
git add modules/video/detector.py
git commit -m "feat: add YOLO player detector"
```

---

### Task 9: Biomechanics Pose Module

**Files:**
- Create: `modules/biomechanics/pose.py`
- Create: `tests/test_biomechanics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_biomechanics.py`:

```python
import numpy as np
import pandas as pd
import pytest
from modules.biomechanics.pose import _angle, calculate_angles, PoseResult
from pathlib import Path


def test_angle_90_degrees():
    # a=(0,0), b=(1,0), c=(1,1) → 90° at b
    result = _angle([0, 0], [1, 0], [1, 1])
    assert abs(result - 90.0) < 0.01


def test_angle_180_degrees():
    # a=(0,0), b=(1,0), c=(2,0) → 180° at b (straight line)
    result = _angle([0, 0], [1, 0], [2, 0])
    assert abs(result - 180.0) < 0.01


def test_angle_45_degrees():
    # a=(0,1), b=(0,0), c=(1,0) → 90°; then diagonal
    result = _angle([1, 0], [0, 0], [0, 1])
    assert abs(result - 90.0) < 0.01


def test_calculate_angles_returns_dataframe():
    # pose_results with no landmarks (None) should still return a DataFrame
    results = [
        PoseResult(frame_path=Path("frame_000001.jpg"), landmarks=None),
        PoseResult(frame_path=Path("frame_000002.jpg"), landmarks=None),
    ]
    df = calculate_angles(results)
    assert isinstance(df, pd.DataFrame)
    assert "frame" in df.columns
    assert len(df) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_biomechanics.py -v
```

Expected: `ImportError` — `pose.py` doesn't exist yet.

- [ ] **Step 3: Create modules/biomechanics/pose.py**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Joint definitions: (proximal, vertex, distal)
JOINTS = {
    "joelho_dir": (
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    ),
    "joelho_esq": (
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
    ),
    "quadril_dir": (
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
    ),
    "quadril_esq": (
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
    ),
    "tornozelo_dir": (
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    ),
    "tornozelo_esq": (
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    ),
}


@dataclass
class PoseResult:
    frame_path: Path
    landmarks: Optional[object]  # mp.solutions.pose.NormalizedLandmarkList or None


def _angle(a: list, b: list, c: list) -> float:
    """Calculate the angle (degrees) at vertex b, given points a, b, c."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product < 1e-8:
        return 0.0
    cosine = np.dot(ba, bc) / norm_product
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def estimate_pose(frame_paths: List[Path]) -> List[PoseResult]:
    """Run MediaPipe Pose on each frame.

    Args:
        frame_paths: List of JPEG frame paths.

    Returns:
        List of PoseResult objects (landmarks=None if detection failed).
    """
    results: List[PoseResult] = []
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img_rgb)
            results.append(PoseResult(frame_path=fp, landmarks=result.pose_landmarks))
    return results


def calculate_angles(pose_results: List[PoseResult]) -> pd.DataFrame:
    """Calculate joint angles for each frame.

    Args:
        pose_results: List of PoseResult from estimate_pose.

    Returns:
        DataFrame with columns: frame, joelho_dir, joelho_esq, quadril_dir,
        quadril_esq, tornozelo_dir, tornozelo_esq.
        NaN where landmarks were not detected.
    """
    rows = []
    for pr in pose_results:
        row: dict = {"frame": pr.frame_path.name}
        if pr.landmarks:
            lm = pr.landmarks.landmark
            for joint_name, (a_lm, b_lm, c_lm) in JOINTS.items():
                a = [lm[a_lm].x, lm[a_lm].y]
                b = [lm[b_lm].x, lm[b_lm].y]
                c = [lm[c_lm].x, lm[c_lm].y]
                row[joint_name] = _angle(a, b, c)
        rows.append(row)
    return pd.DataFrame(rows)


def overlay_skeleton(
    frame_paths: List[Path],
    pose_results: List[PoseResult],
    output_dir: Path,
) -> List[Path]:
    """Draw MediaPipe skeleton overlay on each frame.

    Args:
        frame_paths: Original frame paths.
        pose_results: Pose results aligned with frame_paths.
        output_dir: Directory to save annotated frames.

    Returns:
        List of paths to annotated frame JPEGs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []

    for fp, pr in zip(frame_paths, pose_results):
        img = cv2.imread(str(fp))
        if pr.landmarks:
            mp_drawing.draw_landmarks(img, pr.landmarks, mp_pose.POSE_CONNECTIONS)
        out_path = output_dir / fp.name
        cv2.imwrite(str(out_path), img)
        output_paths.append(out_path)

    return output_paths
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_biomechanics.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/biomechanics/pose.py tests/test_biomechanics.py
git commit -m "feat: add biomechanics pose module (MediaPipe + joint angles)"
```

---

### Task 10: Streamlit Dashboard — Vídeo Page

**Files:**
- Create: `dashboard/pages/2_Video.py`

- [ ] **Step 1: Create dashboard/pages/2_Video.py**

```python
import streamlit as st
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.video.downloader import download, extract_frames
from modules.video.detector import detect_players, annotate_frames

st.set_page_config(page_title="Vídeo", layout="wide")
st.title("🎥 Análise de Vídeo")

st.sidebar.header("Fonte de vídeo")
source = st.sidebar.radio("Origem", ["Upload de arquivo", "URL do YouTube"])

video_path: Path | None = None

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
            dl_dir = Path("data/video_downloads")
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
        frames_dir = Path("data/frames") / video_path.stem
        frames = extract_frames(video_path, start_sec, end_sec, frames_dir)
        st.success(f"{len(frames)} frames extraídos.")

    with st.spinner("Detectando jogadores (YOLO)..."):
        detections = detect_players(frames_dir)
        annotated_dir = Path("data/annotated") / video_path.stem
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
```

- [ ] **Step 2: Test manually**

```bash
streamlit run dashboard/app.py
```

Navegue até **Vídeo**. Faça upload de um MP4 curto. Defina intervalo de 0s a 3s. Clique "Extrair frames e detectar jogadores". Verifique que os frames anotados aparecem.

- [ ] **Step 3: Commit**

```bash
git add dashboard/pages/2_Video.py
git commit -m "feat: add Streamlit Video page with yt-dlp download and YOLO detection"
```

---

### Task 11: Streamlit Dashboard — Biomecânica Page

**Files:**
- Create: `dashboard/pages/3_Biomecanica.py`

- [ ] **Step 1: Create dashboard/pages/3_Biomecanica.py**

```python
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.biomechanics.pose import estimate_pose, calculate_angles, overlay_skeleton
from visualizations.exporter import export_png
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Biomecânica", layout="wide")
st.title("🦴 Biomecânica")

if "frames" not in st.session_state:
    st.info("Primeiro extraia frames na aba **Vídeo**.")
    st.stop()

frame_paths = [Path(p) for p in st.session_state["frames"]]

if st.button("Analisar pose (MediaPipe)"):
    with st.spinner("Estimando pose..."):
        pose_results = estimate_pose(frame_paths)
        skeleton_dir = Path("data/skeleton") / frame_paths[0].parent.name
        overlaid = overlay_skeleton(frame_paths, pose_results, skeleton_dir)
        angles_df = calculate_angles(pose_results)
        st.session_state["pose_results"] = pose_results
        st.session_state["overlaid"] = [str(p) for p in overlaid]
        st.session_state["angles_df"] = angles_df
        st.success("Análise concluída.")

if "angles_df" not in st.session_state:
    st.stop()

angles_df = st.session_state["angles_df"]
overlaid_paths = [Path(p) for p in st.session_state["overlaid"]]

# --- Skeleton frames ---
st.subheader("Frames com esqueleto")
sample = overlaid_paths[::5]
cols = st.columns(min(4, len(sample)))
for i, fp in enumerate(sample[:4]):
    cols[i].image(str(fp), use_container_width=True)

# --- Angle chart ---
st.subheader("Ângulos articulares ao longo do tempo")
joint_cols = [c for c in angles_df.columns if c != "frame"]
selected_joints = st.multiselect("Articulações", joint_cols, default=joint_cols[:2])

fig = go.Figure()
for joint in selected_joints:
    fig.add_trace(go.Scatter(
        x=list(range(len(angles_df))),
        y=angles_df[joint],
        mode="lines+markers",
        name=joint,
    ))
fig.update_layout(
    xaxis_title="Frame",
    yaxis_title="Ângulo (graus)",
    plot_bgcolor="#1a1a2e",
    paper_bgcolor="#1a1a2e",
    font=dict(color="white"),
    legend=dict(bgcolor="#1a1a2e"),
)
st.plotly_chart(fig, use_container_width=True)

# --- Annotations ---
st.subheader("Anotações de momentos-chave")
annotation_text = st.text_area("Anotações (ex: frame 12 = momento do contato com a bola)")
if st.button("Exportar gráfico de ângulos como PNG"):
    mpl_fig, ax = plt.subplots(figsize=(12, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    for joint in selected_joints:
        ax.plot(angles_df[joint].values, label=joint)
    ax.set_xlabel("Frame", color="white")
    ax.set_ylabel("Ângulo (graus)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.set_title("Ângulos Articulares", color="white", fontsize=14)
    plt.tight_layout()
    path = export_png(mpl_fig, "angulos_articulares.png")
    st.success(f"Salvo em {path}")
```

- [ ] **Step 2: Test manually**

```bash
streamlit run dashboard/app.py
```

1. Vá para **Vídeo**, faça upload de um clipe e extraia frames.
2. Vá para **Biomecânica**, clique "Analisar pose".
3. Verifique que os frames com esqueleto aparecem.
4. Verifique que o gráfico de ângulos articulares é renderizado.
5. Clique "Exportar gráfico" e confirme o arquivo em `exports/`.

- [ ] **Step 3: Commit**

```bash
git add dashboard/pages/3_Biomecanica.py
git commit -m "feat: add Streamlit Biomecânica page with pose overlay and angle charts"
```

---

### Task 12: Final Run — All Tests Pass

- [ ] **Step 1: Run full test suite**

```bash
cd ~/projects/podcast-analytics
pytest tests/ -v
```

Expected output:
```
tests/test_statsbomb.py::test_list_competitions_returns_dataframe PASSED
tests/test_statsbomb.py::test_list_matches_passes_ids PASSED
tests/test_statsbomb.py::test_get_events_returns_dataframe PASSED
tests/test_statsbomb.py::test_get_360_returns_dataframe PASSED
tests/test_positioning.py::test_heatmap_returns_figure PASSED
tests/test_positioning.py::test_pass_map_returns_figure PASSED
tests/test_positioning.py::test_pressure_map_returns_figure PASSED
tests/test_positioning.py::test_avg_positions_returns_figure PASSED
tests/test_exporter.py::test_export_png_creates_file PASSED
tests/test_exporter.py::test_export_png_returns_path PASSED
tests/test_video.py::test_extract_frames_creates_jpegs PASSED
tests/test_video.py::test_extract_frames_respects_time_range PASSED
tests/test_biomechanics.py::test_angle_90_degrees PASSED
tests/test_biomechanics.py::test_angle_180_degrees PASSED
tests/test_biomechanics.py::test_angle_45_degrees PASSED
tests/test_biomechanics.py::test_calculate_angles_returns_dataframe PASSED
16 passed
```

- [ ] **Step 2: Final commit**

```bash
git add .
git commit -m "chore: verified all 16 tests pass — Phase 1 + Phase 2 complete"
```

---

## Known Limitations

- **Futebol brasileiro:** dados estruturados gratuitos são escassos. Para episódios sobre o Brasileirão, usar o pipeline de Vídeo + Biomecânica; o pipeline de Posicionamento funciona melhor para ligas europeias disponíveis no StatsBomb Open Data.
- **YOLO em baixa resolução:** se o vídeo tiver qualidade ruim, as bounding boxes podem ser imprecisas. O usuário pode selecionar manualmente a ROI (região de interesse) em uma versão futura.
- **MediaPipe em oclusões:** quando um jogador está coberto por outro nos frames, a pose pode falhar. Os frames problemáticos geram `landmarks=None` e aparecem sem esqueleto no dashboard.
