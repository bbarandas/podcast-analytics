from pathlib import Path
from typing import List
import cv2
from yt_dlp import YoutubeDL


def download(url: str, output_dir: Path) -> Path:
    """Download a YouTube video as MP4 to output_dir.

    Args:
        url: YouTube URL.
        output_dir: Directory to save the downloaded file.

    Returns:
        Path to the downloaded MP4 file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "mp4",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "quiet": True,
        "cookiesfrombrowser": ("chrome",),
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    mp4_files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4_files:
        raise FileNotFoundError(f"No MP4 file found in {output_dir} after download.")
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
