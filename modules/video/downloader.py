import subprocess
from pathlib import Path
from typing import List
from yt_dlp import YoutubeDL


def download(url: str, output_dir: Path) -> Path:
    """Download a YouTube video as MP4 to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "mp4/best",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["ios"]}},
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
    """Extract frames from a video between start_sec and end_sec using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "frame_%06d.jpg")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", str(video_path),
            "-q:v", "2",
            output_pattern,
        ],
        check=True,
        capture_output=True,
    )
    return sorted(output_dir.glob("*.jpg"))
