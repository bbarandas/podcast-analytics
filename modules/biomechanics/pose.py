from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

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
            if img is None:
                results.append(PoseResult(frame_path=fp, landmarks=None))
                continue
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
        if img is None:
            continue
        if pr.landmarks:
            mp_drawing.draw_landmarks(img, pr.landmarks, mp_pose.POSE_CONNECTIONS)
        out_path = output_dir / fp.name
        cv2.imwrite(str(out_path), img)
        output_paths.append(out_path)

    return output_paths
