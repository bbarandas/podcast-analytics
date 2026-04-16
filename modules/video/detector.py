from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import cv2
from ultralytics import YOLO


@dataclass
class Detection:
    frame_path: Path
    # Each box: [x1, y1, x2, y2, confidence, class_id]
    boxes: List[List[float]] = field(default_factory=list)


def detect_players(frames_dir: Path, model_name: str = "yolov8n.pt") -> List[Detection]:
    """Run YOLO person detection on all JPEGs in frames_dir.

    Args:
        frames_dir: Directory containing frame JPEGs.
        model_name: YOLOv8 model variant (yolov8n.pt = nano, fastest).

    Returns:
        List of Detection objects, one per frame.
    """
    model = YOLO(model_name)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    detections: List[Detection] = []

    for frame_path in frame_paths:
        results = model(str(frame_path), classes=[0], verbose=False)  # class 0 = person
        boxes = results[0].boxes.data.tolist() if results[0].boxes is not None else []
        detections.append(Detection(frame_path=frame_path, boxes=boxes))

    return detections


def annotate_frames(detections: List[Detection], output_dir: Path) -> List[Path]:
    """Draw bounding boxes on frames and save to output_dir.

    Args:
        detections: List of Detection objects.
        output_dir: Directory to save annotated JPEGs.

    Returns:
        List of paths to annotated frame JPEGs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []

    for det in detections:
        frame = cv2.imread(str(det.frame_path))
        for i, box in enumerate(det.boxes):
            x1, y1, x2, y2, conf, _ = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame, f"#{i} {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        out_path = output_dir / det.frame_path.name
        cv2.imwrite(str(out_path), frame)
        output_paths.append(out_path)

    return output_paths
