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
