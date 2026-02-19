#!/usr/bin/env python3
"""Train ReLU YOLO models and export ONNX, plus export YOLO-World v1/v2 ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def trained_weight_path(task: str, run_name: str) -> Path:
    """Return expected trained weight path for a run."""
    return Path(f"runs/{task}/runs/relu_train/{run_name}/weights/best.pt")


def _world_v1_export_sizes(imgsz: int) -> list[int]:
    """Return robust fallback sizes for YOLO-World v1 ONNX export.

    NOTE:
    YOLO-World v1 ONNX export may fail on some sizes (e.g. 640/1280) due to adaptive
    pooling export constraints. Empirically, sizes divisible by 96 are reliable.
    """
    base = int(imgsz)
    down = max(96, (base // 96) * 96)
    up = max(96, ((base + 95) // 96) * 96)

    candidates = []
    if base % 96 == 0:
        candidates.append(base)

    # Prefer closest compatible sizes first, then known safe fallbacks.
    nearest = sorted({down, up}, key=lambda x: (abs(x - base), x))
    candidates.extend(nearest)
    candidates.extend([960, 1248, 1344, 640, 96])

    ordered = []
    for x in candidates:
        if x > 0 and x not in ordered:
            ordered.append(x)
    return ordered


def train_and_export(
    task: str,
    model_cfg: str,
    data: str,
    run_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    skip_train_if_exists: bool,
    opset: int | None = None,
) -> dict:
    best_pt = trained_weight_path(task, run_name)

    if not (skip_train_if_exists and best_pt.exists()):
        model = YOLO(model_cfg)
        results = model.train(
            task=task,
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            project=str(Path.cwd() / "runs" / "relu_train"),
            name=run_name,
            pretrained=False,
        )
        save_dir = Path(results.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
    else:
        save_dir = best_pt.parents[1]

    if not best_pt.exists():
        best_pt = save_dir / "weights" / "last.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"No trained weight found for {run_name} in {save_dir / 'weights'}")

    export_model = YOLO(str(best_pt))
    onnx_path = export_model.export(
        format="onnx",
        imgsz=imgsz,
        device=device,
        simplify=False,
        opset=opset,
        project=str(Path.cwd() / "runs" / "onnx_export"),
        name=run_name,
    )
    # Normalize ReLU export filename to run_name.onnx (avoid ambiguous "best.onnx")
    onnx_path = Path(str(onnx_path))
    target_onnx = onnx_path.parent / f"{run_name}.onnx"
    if onnx_path != target_onnx and onnx_path.exists():
        if target_onnx.exists():
            target_onnx.unlink()
        onnx_path.replace(target_onnx)
        onnx_path = target_onnx

    return {"run_name": run_name, "best_pt": str(best_pt), "onnx": str(onnx_path)}

def export_world(weights: str, run_name: str, imgsz: int, device: str, opset: int | None = None) -> dict:
    is_world_v1 = ("-world" in weights) and ("worldv2" not in weights)

    last_error = None
    for size in _world_v1_export_sizes(imgsz):
        try:
            print(f"[RUN] {run_name}: trying YOLO-World v1 export with imgsz={size}")
            model = YOLO(weights)
            onnx_path = model.export(
                format="onnx",
                imgsz=size,
                device=device,
                simplify=False,
                opset=opset,
                project=str(Path.cwd() / "runs" / "onnx_export"),
                name=run_name,
            )
            # Normalize export filename to run_name.onnx (avoid ambiguous "best.onnx")
            onnx_path = Path(str(onnx_path))
            target_onnx = onnx_path.parent / f"{run_name}.onnx"
            if onnx_path != target_onnx and onnx_path.exists():
                if target_onnx.exists():
                    target_onnx.unlink()
                onnx_path.replace(target_onnx)
                onnx_path = target_onnx
            return {"run_name": run_name, "weights": weights, "imgsz": size, "onnx": str(onnx_path)}
        except Exception as e:
            last_error = e
            if not is_world_v1:
                raise RuntimeError(f"YOLO-World v2 ONNX export failed. Last error: {last_error}")

    raise RuntimeError(f"YOLO-World v1 ONNX export failed for all tried sizes. Last error: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--world-imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-trained", action="store_true")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version (default: auto)")
    args = parser.parse_args()

    relu_jobs = [
        ("detect", "ultralytics/cfg/models/v5/yolov5-reluact.yaml", "coco8.yaml", "yolov5-relu"),
        ("detect", "ultralytics/cfg/models/v6/yolov6-relu.yaml", "coco8.yaml", "yolov6-relu"),
        ("detect", "ultralytics/cfg/models/v8/yolov8-relu.yaml", "coco8.yaml", "yolov8-relu"),
        ("detect", "ultralytics/cfg/models/11/yolo11-relu.yaml", "coco8.yaml", "yolo11-relu"),
        ("detect", "ultralytics/cfg/models/26/yolo26-relu.yaml", "coco8.yaml", "yolo26-relu"),
        ("segment", "ultralytics/cfg/models/11/yolo11-seg-relu.yaml", "coco8-seg.yaml", "yolo11-seg-relu"),
        ("segment", "ultralytics/cfg/models/26/yolo26-seg-relu.yaml", "coco8-seg.yaml", "yolo26-seg-relu"),
        ("pose", "ultralytics/cfg/models/11/yolo11-pose-relu.yaml", "coco8-pose.yaml", "yolo11-pose-relu"),
        ("pose", "ultralytics/cfg/models/26/yolo26-pose-relu.yaml", "coco8-pose.yaml", "yolo26-pose-relu"),
    ]
    world_jobs = [
        ("yolov8s-world", "yoloworld-v1"),
        ("yolov8s-worldv2", "yoloworld-v2"),
    ]

    summary = {"relu_train_export": [], "world_export": []}

    for task, cfg, data, run_name in relu_jobs:
        try:
            print(f"[RUN] train+export {run_name} ({task})")
            summary["relu_train_export"].append(
                train_and_export(
                    task=task,
                    model_cfg=cfg,
                    data=data,
                    run_name=run_name,
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    workers=args.workers,
                    skip_train_if_exists=args.skip_trained,
                    opset=args.opset,
                )
            )
        except Exception as e:
            summary["relu_train_export"].append({"run_name": run_name, "error": str(e)})
            print(f"[ERROR] {run_name}: {e}")

    for weights, run_name in world_jobs:
        try:
            print(f"[RUN] export {run_name} ({weights})")
            summary["world_export"].append(
                export_world(weights=weights, run_name=run_name, imgsz=args.world_imgsz, device=args.device, opset=args.opset)
            )
        except Exception as e:
            summary["world_export"].append({"run_name": run_name, "weights": weights, "error": str(e)})
            print(f"[ERROR] {run_name}: {e}")

    out = Path("runs/relu_train_export_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] Summary saved to {out}")


if __name__ == "__main__":
    main()
