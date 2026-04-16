import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent

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
        skeleton_dir = PROJECT_ROOT / "data" / "skeleton" / frame_paths[0].parent.name
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
annotation_text = st.text_area(
    "Anotações (ex: frame 12 = momento do contato com a bola)",
    value=st.session_state.get("annotation", ""),
)
st.session_state["annotation"] = annotation_text
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
