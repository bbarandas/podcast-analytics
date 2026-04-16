import pandas as pd
from matplotlib.figure import Figure
from mplsoccer import Pitch
from visualizations.theme import (
    PITCH_BG, LINE_COLOR, ACCENT_GREEN, ACCENT_RED,
    ACCENT_BLUE, TEXT_COLOR, FONT_SIZE_TITLE, FONT_SIZE_LABEL
)


def _base_pitch() -> tuple:
    """Return (Pitch, fig, ax) with podcast theme applied."""
    # pitch_color="grass" is intentional: green field on a dark navy figure background.
    # PITCH_BG is applied to the figure face (outer area), not the pitch surface.
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

    # Split into successful (NaN outcome) and failed passes
    successful = passes[pd.isna(passes["pass_outcome"])]
    failed = passes[~pd.isna(passes["pass_outcome"])]

    for group, color in [(successful, ACCENT_GREEN), (failed, ACCENT_RED)]:
        if len(group) > 0:
            xs = [loc[0] for loc in group["location"]]
            ys = [loc[1] for loc in group["location"]]
            exs = [loc[0] for loc in group["pass_end_location"]]
            eys = [loc[1] for loc in group["pass_end_location"]]
            pitch.arrows(xs, ys, exs, eys, ax=ax, color=color, width=1.5, headwidth=5, zorder=2)

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
