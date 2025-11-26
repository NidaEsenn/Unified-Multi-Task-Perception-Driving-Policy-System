# Unified Multi-Task Perception & Driving Policy System

Lightweight research and portfolio scaffold for autonomous driving tasks: vehicle detection, lane and drivable-area segmentation, policy learning (ConvLSTM), and realtime overlay demos. This repository provides simple placeholders to quickly iterate and integrate models and datasets.
<img width="1216" height="703" alt="Screenshot 2025-11-26 at 12 01 51 AM" src="https://github.com/user-attachments/assets/b6911fde-67f7-4633-9068-1b6cba5d2c54" />

## Project layout

- `perception/` - vehicle detection and segmentation model scaffolds.
- `policy/` - policy models and training/eval scripts.
- `data_engine/` - utilities for extracting frames and basic dataset metrics.
- `realtime_demo/` - overlay demo that composes detectors and masks for visualization.
- `utils/` - configuration and dataset helpers.
- `tests/` - lightweight smoke tests for CI.
- `outputs/` - generated demo videos and visualizations (ignored by git)
- `.vscode/` - optional local workspace settings (ignored by git)
- `requirements.txt` - runtime dependencies.
- `requirements-dev.txt` - development & CI dependencies (includes PyTorch for expanded tests).

## Getting started

1) Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Run a quick smoke test locally (minimal dependencies are installed above):

```bash
python -m tests.smoke_test
```

3) Optional: install advanced dependencies (PyTorch / dev) for full tests and training:

```bash
# On Linux, install CPU-only PyTorch wheel (choose version as needed). Example below uses the official CPU wheel index:
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements-dev.txt
```

## Quick examples

- Run the overlay demo with a webcam:
```bash
python -m realtime_demo.overlay_demo --source 0 --weights-det yolov8n.pt
```

- Train the policy (placeholder):
```bash
python -m policy.train_policy --data-dir data/train --epochs 10 --batch-size 8 --lr 1e-3
```

### Balanced training (sample)
To improve performance on rarer steering values and reduce collapse-to-zero steering, run the balanced training helper script which uses a WeightedRandomSampler binned on steering labels.

```bash
python scripts/train_balanced.py --epochs 8 --batch-size 16 --num-workers 2
```

This script stores the balanced best checkpoint at `checkpoints/policy/policy_balanced_best.pt`.

### Real-time demo outputs
The demo writes annotated overlay videos to an `outputs/` directory by default. These files are generated artifacts and are ignored by Git (see `.gitignore`).

Example (generate a short demo and save it to outputs/demo_output.mp4):
```bash
python -m realtime_demo.overlay_demo \
  --input-video data/raw_videos/my_drive.mp4 \
  --output-video outputs/demo_output.mp4 \
  --seq-len 3 --device cpu --max-frames 200
```
The recorded `outputs/demo_output.mp4` contains bounding boxes for detected vehicles, lane and drivable-area mask overlays, and a small steering HUD.

### Demo GIF / Snapshot (Optional)
To include a short animated preview in documentation, convert a small portion of the demo video into a compact GIF:

```bash
# create a short 5-second GIF at 10 FPS and width 640px
ffmpeg -ss 00:00:02 -t 5 -i outputs/demo_output.mp4 -vf "scale=640:-1,fps=10" -y docs/demo_overlay.gif
```

Then add the resulting GIF at `docs/demo_overlay.gif` and include it in the README to preview overlays:

```markdown
![Demo Overlay](docs/demo_overlay.gif)
```

Tip: Keep GIF duration and FPS small to avoid bloating the repository; use `outputs/` for generated videos and place only committed demo artifacts in `docs/` or `assets/`.

### Example: generate demo with balanced policy model
If you've run the balanced training script and produced `checkpoints/policy/policy_balanced_best.pt`, you can generate a demo that overlays the policy's steering predictions on a sample video:

```bash
# generate a short demo using the balanced checkpoint and save to `outputs/realtime_demo_test2.mp4`
PYTHONPATH=. python -m realtime_demo.overlay_demo \
  --input-video data/raw_videos/YTDown.com_YouTube_APEMAN-s-Star-Product-C450-Series-A-Dash_Media_4LdYRwgsfLM_002_720p.mp4 \
  --output-video outputs/realtime_demo_test2.mp4 \
  --weights-policy checkpoints/policy/policy_balanced_best.pt \
  --seq-len 3 --max-frames 200 --debug-policy
```

The output `outputs/realtime_demo_test2.mp4` contains bounding boxes, lane and drivable-area overlays, and a steering HUD showing predicted steering values. Use `--debug-policy` to print steering tensor values to the console while the demo runs.

## Contributing

1. Fork the repo, create a feature branch and open a PR.
2. Add tests and documentation for new components.
3. Use the CI workflow as a baseline: quick smoke checks on PRs and optional expanded tests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
# Unified-Multi-Task-Perception-Driving-Policy-System
A lightweight student-built autonomous driving foundation stack featuring multi-task perception (YOLOv8 + U-Net), a temporal end-to-end driving policy network (ConvLSTM), and a data-engine pipeline for frame curation and real-time evaluation.
# 🚗 Autonomous Driving Foundation Stack (Student Edition)
<img width="1536" height="1024" alt="Autonomous Driving Foundation Stack Design" src="https://github.com/user-attachments/assets/76af247e-8548-42a9-85e8-8a549ec37210" />

A lightweight autonomous driving foundation stack designed for research and educational purposes.  
This project demonstrates a unified approach to **multi-task perception**, **temporal end-to-end control**, and **data-engine curation**—all core components used in modern self-driving systems.

Built using **Python, PyTorch, OpenCV, YOLOv8, U-Net, and ConvLSTM**, this stack provides a clean foundation for students exploring perception, planning, and imitation-learning–based driving policies.

---

## 📌 Key Features

### 🧩 Multi-Task Perception Module
- **Vehicle Detection** using YOLOv8  
- **Lane Segmentation** using a U-Net architecture  
- **Drivable Area Prediction** for scene understanding  
- Shared encoder design for efficient multi-task learning  
- Generates frame-by-frame perception overlays  

### 🎮 End-to-End Driving Policy (Imitation Learning)
- Temporal model built with **ConvLSTM**  
- Predicts **steering commands** from sequential image frames  
- Trained on curated driving datasets  
- Incorporates motion/temporal patterns absent in single-frame models  

### 🧠 Data Engine — Frame Curation Pipeline
- Filters **blurry**, **overexposed**, and **low-information** frames  
- Edge-case mining using:
  - Steering change magnitude  
  - Frame-to-frame motion  
  - Brightness/contrast scores  
- Outputs a curated subset for high-quality model training  

### 🎥 Real-Time Demo System
- Overlays:
  - detected vehicles  
  - lane boundaries  
  - drivable area mask  
  - predicted steering angle  
- Rendered as a single annotated driving video for evaluation  

---

## 🏗️ Architecture Overview
                    +--------------------------+
                    |      Input Frame(s)      |
                    +-------------+------------+
                                  |
            +---------------------+--------------------+
            |                                          |
    +-------v--------+                        +--------v---------+
    |  Perception     |                        |  Driving Policy  |
    | (YOLO + U-Net)  |                        |   (ConvLSTM)     |
    +-------+--------+                        +--------+---------+
            |                                          |
            +---------------------+--------------------+
                                  |
                    +-------------v------------+
                    |   Real-Time Visualizer   |
                    +--------------------------+

---

## 📂 Repository Structure

Autonomous-Driving-Foundation-Stack/
│
├── perception/
│ ├── vehicle_detection_yolov8.py
│ ├── lane_unet.py
│ ├── drivable_area_unet.py
│
├── policy/
│ ├── convlstm_model.py
│ ├── train_policy.py
│ ├── evaluate_policy.py
│
├── data_engine/
│ ├── curate_frames.py
│ ├── quality_metrics.py # blur, motion, brightness
│
├── realtime_demo/
│ ├── overlay_demo.py
│
├── data/
│ └── raw_videos/
│
├── results/
│ ├── demo_output.mp4
│ ├── sample_overlays/
│
└── README.md


---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt

### 2. Run Perception Module
python perception/vehicle_detection_yolov8.py
python perception/lane_unet.py

### 3. Curate Dataset With Data Engine
python data_engine/curate_frames.py --input data/raw_videos/video.mp4
### 4. Train End-to-End Driving Policy
python policy/train_policy.py --data curated/
### 5. Generate Real-Time Demo Output
python realtime_demo/overlay_demo.py



🎯 Future Improvements
Add BEV (Bird’s-Eye View) transformation
Introduce reinforcement learning for fine-tuned control
Fuse multiple sensors (additional cameras or pseudo-LiDAR)
Integrate lightweight transformer-based temporal models
🙌 Acknowledgements
This project was built as a student-oriented foundation stack to explore core concepts behind modern autonomous driving systems and multi-task perception networks.




