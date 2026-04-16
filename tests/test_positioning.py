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
