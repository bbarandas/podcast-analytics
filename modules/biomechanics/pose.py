from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Joint definitions: (proximal, vertex, distal) using PoseLandmark enum
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
    """Run MediaPipe Pose on each frame."""
    results: List[PoseResult] = []
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        for fp in frame_paths:
            try:
                img_rgb = np.array(Image.open(str(fp)).convert("RGB"))
            except Exception:
                results.append(PoseResult(frame_path=fp, landmarks=None))
                continue
            result = pose.process(img_rgb)
            results.append(PoseResult(frame_path=fp, landmarks=result.pose_landmarks))
    return results


def calculate_angles(pose_results: List[PoseResult]) -> pd.DataFrame:
    """Calculate joint angles for each frame."""
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
    """Draw MediaPipe skeleton overlay on each frame."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []

    for fp, pr in zip(frame_paths, pose_results):
        try:
            img_rgb = np.array(Image.open(str(fp)).convert("RGB"))
        except Exception:
            continue
        if pr.landmarks:
            mp_drawing.draw_landmarks(img_rgb, pr.landmarks, mp_pose.POSE_CONNECTIONS)
        Image.fromarray(img_rgb).save(str(output_dir / fp.name))
        output_paths.append(output_dir / fp.name)

    return output_paths
