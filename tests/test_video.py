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
