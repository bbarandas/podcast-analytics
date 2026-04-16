from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import cv2
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage
import numpy as np
import pandas as pd

# Joint definitions: (proximal, vertex, distal) using PoseLandmark indices
JOINTS = {
    "joelho_dir": (11, 13, 15),  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
    "joelho_esq": (12, 14, 16),  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
    "quadril_dir": (5, 11, 13),  # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE
    "quadril_esq": (6, 12, 14),  # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
    "tornozelo_dir": (13, 15, 17),  # RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX
    "tornozelo_esq": (14, 16, 18),  # LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX
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
    """Run MediaPipe PoseLandmarker on each frame.

    Args:
        frame_paths: List of JPEG frame paths.

    Returns:
        List of PoseResult objects (landmarks=None if detection failed).
    """
    results: List[PoseResult] = []
    options = vision.PoseLandmarkerOptions(
        base_options=vision.BaseOptions(model_asset_path=None),
        running_mode=vision.RunningMode.IMAGE,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=MPImage.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            results.append(PoseResult(frame_path=fp, landmarks=landmarks))
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
            lm = pr.landmarks
            for joint_name, (a_idx, b_idx, c_idx) in JOINTS.items():
                a = [lm[a_idx].x, lm[a_idx].y]
                b = [lm[b_idx].x, lm[b_idx].y]
                c = [lm[c_idx].x, lm[c_idx].y]
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
        if pr.landmarks:
            vision.drawing_utils.draw_landmarks(
                img, pr.landmarks, vision.PoseLandmarksConnections.POSE_CONNECTIONS
            )
        out_path = output_dir / fp.name
        cv2.imwrite(str(out_path), img)
        output_paths.append(out_path)

    return output_paths
