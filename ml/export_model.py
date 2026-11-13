"""
Downloads pretrained YOLOv8n weights and exports a static-shape [1, 3, 640, 640]
ONNX model directly into `public/models/yolov8n.onnx` for browser inference.

Usage:
    pip install -r ml/requirements.txt
    python ml/export_model.py
"""

from pathlib import Path
import shutil
from ultralytics import YOLO


def export_yolov8_to_onnx() -> None:
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "public" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print("[2/3] Exporting static ONNX graph (opset=12, imgsz=640)...")
    exported_path = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=False,
        dynamic=False,
    )

    target_path = models_dir / "yolov8n.onnx"
    shutil.move(str(exported_path), str(target_path))
    size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"[3/3] Successfully exported ONNX model to: {target_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    export_yolov8_to_onnx()