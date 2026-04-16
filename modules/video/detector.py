from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    frame_path: Path
    # Each box: [x1, y1, x2, y2, confidence, class_id]
    boxes: List[List[float]] = field(default_factory=list)


def detect_players(frames_dir: Path, model_name: str = "yolov8n.pt") -> List[Detection]:
    """Run YOLO person detection on all JPEGs in frames_dir."""
    model = YOLO(model_name)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    detections: List[Detection] = []

    for frame_path in frame_paths:
        img = np.array(Image.open(str(frame_path)))
        results = model(img, classes=[0], verbose=False)  # class 0 = person
        boxes = results[0].boxes.data.tolist() if results[0].boxes is not None else []
        detections.append(Detection(frame_path=frame_path, boxes=boxes))

    return detections


def annotate_frames(detections: List[Detection], output_dir: Path) -> List[Path]:
    """Draw bounding boxes on frames and save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []

    for det in detections:
        img = Image.open(str(det.frame_path))
        draw = ImageDraw.Draw(img)
        for i, box in enumerate(det.boxes):
            x1, y1, x2, y2, conf, _ = box
            draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=(0, 255, 0), width=2)
            draw.text((int(x1), int(y1) - 15), f"#{i} {conf:.2f}", fill=(0, 255, 0))
        out_path = output_dir / det.frame_path.name
        img.save(str(out_path))
        output_paths.append(out_path)

    return output_paths
