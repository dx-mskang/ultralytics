# ReLU YOLO -> ONNX Quick README (KO)

이 문서는 **복붙해서 바로 실행**하는 용도입니다.
목표:

- ReLU 학습: `v5/v6/v8/v11/v26` detect, `v11/v26` seg/pose
- ONNX export: 위 모델 + YOLO-World `v1/v2`

## 0) 환경 준비

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install "onnx>=1.12.0,<2.0.0"
```

## 1) 한 번에 학습 + ONNX export (권장)

```bash
.venv/bin/python scripts/train_export_pipeline.py \
  --epochs 100 \
  --imgsz 640 \
  --world-imgsz 1280 \
  --batch 4 \
  --device cpu \
  --workers 0 \
  --opset 21  # ONNX opset 21로 설정
```

빠른 테스트(스모크):

```bash
.venv/bin/python scripts/train_export_pipeline.py \
  --epochs 1 \
  --imgsz 96 \
  --batch 4 \
  --device cpu \
  --workers 0 \
  --opset 21  # ONNX opset 21로 설정
```

이미 학습된 ReLU `best.pt` 재사용해서 ONNX만 다시 export:

```bash
.venv/bin/python scripts/train_export_pipeline.py \
  --skip-trained \
  --imgsz 640 \
  --world-imgsz 1280 \
  --batch 4 \
  --device cpu \
  --workers 0 \
  --opset 21  # ONNX opset 21로 설정
```

## 2) 커맨드 실행 시 생성 파일/경로

`train_export_pipeline.py` 실행 후 기본적으로 아래가 생성됩니다.

- 공통 요약 파일:
  - `runs/relu_train_export_summary.json`
- ReLU 학습/Export 결과:
  - Detect: `runs/detect/runs/relu_train/<run_name>/`
  - Segment: `runs/segment/runs/relu_train/<run_name>/`
  - Pose: `runs/pose/runs/relu_train/<run_name>/`
- 각 `<run_name>` 폴더의 주요 파일:
  - `args.yaml` (실행 파라미터)
  - `results.csv`, `results.png` (학습 결과)
  - `weights/last.pt` (마지막 epoch)
  - `weights/best.pt` (최고 성능 checkpoint)
  - `weights/<run_name>.onnx` (ONNX 결과, 예: `weights/yolov8-relu.onnx`)
- YOLO-World Export 결과:
  - `yolov8s-world.onnx` (v1)
  - `yolov8s-worldv2.onnx` (v2)
  - 필요 시 가중치 파일 `yolov8s-world.pt`, `yolov8s-worldv2.pt`가 작업 디렉터리에 생성될 수 있음

실제 run 이름 예시:

- detect: `yolov5-relu`, `yolov6-relu`, `yolov8-relu`, `yolo11-relu`, `yolo26-relu`
- segment: `yolo11-seg-relu`, `yolo26-seg-relu`
- pose: `yolo11-pose-relu`, `yolo26-pose-relu`

## 3) 결과 확인 커맨드

요약 JSON:

```bash
cat runs/relu_train_export_summary.json
```

ReLU 학습 산출물(`best.pt`, `<run_name>.onnx`) 확인:

```bash
ls runs/detect/runs/relu_train/*/weights
ls runs/segment/runs/relu_train/*/weights
ls runs/pose/runs/relu_train/*/weights
```

YOLO-World ONNX 확인:

```bash
ls -l yolov8s-world*.onnx
```

## 4) `best.pt`가 생기는 이유 (짧게)

- `best.pt`는 **원래 있던 파일이 아니라 학습 중 생성**됩니다.
- 저장 위치: `runs/<task>/runs/relu_train/<run_name>/weights/best.pt`
- 학습 중 가장 좋은 검증 점수(fitness)일 때 `best.pt`로 저장됩니다.
- 참고 코드: `ultralytics/engine/trainer.py:149`, `ultralytics/engine/trainer.py:659`, `ultralytics/engine/trainer.py:660`

## 5) ReLU YAML 최소 규칙

기존 YAML 복사 후 `nc:` 아래 1줄 추가:

```yaml
activation: torch.nn.ReLU() # (optional) model default activation function
```

이미 만들어둔 ReLU YAML:

- `ultralytics/cfg/models/v5/yolov5-reluact.yaml`
- `ultralytics/cfg/models/v6/yolov6-relu.yaml`
- `ultralytics/cfg/models/v8/yolov8-relu.yaml`
- `ultralytics/cfg/models/11/yolo11-relu.yaml`
- `ultralytics/cfg/models/26/yolo26-relu.yaml`
- `ultralytics/cfg/models/11/yolo11-seg-relu.yaml`
- `ultralytics/cfg/models/26/yolo26-seg-relu.yaml`
- `ultralytics/cfg/models/11/yolo11-pose-relu.yaml`
- `ultralytics/cfg/models/26/yolo26-pose-relu.yaml`

## 6) ONNX opset 설정

- 기본값: 자동 선택 (Torch 버전에 따라 결정)
- opset 21로 명시적 설정: `--opset 21` 인자 추가
- 지원되는 opset 범위: ONNX 버전에 따라 다름

## 7) YOLO-World v1/v2 메모

- `v2`: `imgsz=1280`로 export
- `v1`: 일부 환경에서 `1280` 실패 가능, 스크립트가 호환 해상도(예: `1248`)로 자동 재시도
- 결과 파일은 요약 JSON의 `world_export` 섹션에서 최종 `imgsz` 확인
