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
    # a=(1,0), b=(0,0), c=(0,1) → 90° at b
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
