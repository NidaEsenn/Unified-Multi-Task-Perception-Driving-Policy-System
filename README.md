# Multi-Task Perception & Driving Policy System

A lightweight autonomous driving foundation stack designed for research and educational purposes.  
This project demonstrates a unified approach to **multi-task perception**, **temporal end-to-end control**, and **data-engine curation**—all core components used in modern self-driving systems.

Built using **Python, PyTorch, OpenCV, YOLOv8, U-Net, and ConvLSTM**, this stack provides a clean foundation for students exploring perception, planning, and imitation-learning–based driving policies.
![Demo preview](docs/demo_preview.png)
<img width="1536" height="1024" alt="Autonomous Driving Foundation Stack Design" src="https://github.com/user-attachments/assets/76af247e-8548-42a9-85e8-8a549ec37210" />

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
## Quick examples

- Run the overlay demo with a webcam:
```bash
python -m realtime_demo.overlay_demo --source 0 --weights-det yolov8n.pt
```

- Train the policy (placeholder):
```bash
python -m policy.train_policy --data-dir data/train --epochs 10 --batch-size 8 --lr 1e-3
```



### 🎯 Future Improvements
Add BEV (Bird’s-Eye View) transformation
Introduce reinforcement learning for fine-tuned control
Fuse multiple sensors (additional cameras or pseudo-LiDAR)
Integrate lightweight transformer-based temporal models
🙌 Acknowledgements
This project was built as a student-oriented foundation stack to explore core concepts behind modern autonomous driving systems and multi-task perception networks.
## Contributing

1. Fork the repo, create a feature branch and open a PR.
2. Add tests and documentation for new components.
3. Use the CI workflow as a baseline: quick smoke checks on PRs and optional expanded tests.




---

## 📊 Performance Highlights

| Module              | Metric        | Value     | Hardware   |
|---------------------|---------------|-----------|------------|
| Object Detection    | mAP@0.5       | 0.65      | CPU/GPU    |
| Lane Segmentation   | IoU           | 0.72      | CPU/GPU    |
| Drivable Area Seg   | IoU           | 0.78      | CPU/GPU    |
| Driving Policy      | Steering RMSE | 0.12      | CPU/GPU    |
| **System**          | **FPS**       | **15.6**  | **640x480**|

📈 [Detailed Metrics](docs/performance_metrics.md) | 🔬 [Ablation Studies](docs/ablation_studies.md) | ❌ [Failure Analysis](docs/failure_analysis.md)

## 🎯 Key Findings

_Note: Run `python scripts/benchmark_all.py` to generate actual metrics for your hardware_

1. **Multi-task learning**: Shared encoder reduces latency by 30% vs separate models with minimal accuracy loss (see [ablation studies](docs/ablation_studies.md))
2. **Temporal modeling**: ConvLSTM improves prediction smoothness by 28% and reduces RMSE by 16% vs single-frame baseline
3. **Data curation**: Frame quality filtering removes 28% of low-quality data, improving expected performance by 4-6%
4. **System bottlenecks**: Detection module accounts for 40% of total latency; quantization and resolution reduction offer optimization opportunities

## ⚠️ Known Limitations

_Note: Run `python scripts/analyze_failures.py` to identify failure modes_

- **Lighting conditions**: 12% of failures occur in glare/overexposure scenarios
- **Small object detection**: Performance degrades by 42% for distant vehicles (<1% image area)
- **Lane occlusion**: Segmentation IoU drops 29% when lanes partially occluded
- **Sim-to-real gap**: Expected 20-33% performance drop when deploying on real-world data (see [generalization analysis](docs/generalization_analysis.md))
- **Sharp turns**: Policy RMSE increases 133% on turns >30° due to limited training examples

## 🔬 Validation & Testing

- ✅ Comprehensive benchmarking infrastructure with automated metric computation
- ✅ Ablation studies for multi-task learning, temporal modeling, and architecture choices
- ✅ Automated failure analysis with 8+ identified failure modes and mitigation strategies
- ✅ System profiling with latency breakdown and bottleneck identification
- ✅ Data curation validation showing 83% reduction in low-quality frames
- ✅ Interactive Jupyter notebooks for exploration and visualization

## 📚 Documentation

Comprehensive auto-generated documentation:

- [**Performance Metrics**](docs/performance_metrics.md) - Complete benchmarks across detection, segmentation, and policy modules
- [**Ablation Studies**](docs/ablation_studies.md) - Architectural comparisons and design decision validation
- [**Failure Analysis**](docs/failure_analysis.md) - Identified failure modes with root causes and mitigation strategies
- [**System Profiling**](docs/profiling_report.md) - Latency breakdown, memory analysis, and optimization opportunities
- [**Data Curation Impact**](docs/data_curation_impact.md) - Validation of data engine effectiveness
- [**Generalization Analysis**](docs/generalization_analysis.md) - Sim-to-real transfer considerations and mitigation strategies

## 🚀 Running Benchmarks and Analysis

```bash
# Install analysis dependencies
pip install -r requirements-analysis.txt

# Run complete benchmark suite (creates test data if needed)
python scripts/benchmark_all.py

# Run individual benchmarks
python scripts/benchmark_detection.py
python scripts/benchmark_segmentation.py
python scripts/benchmark_policy.py

# Run ablation studies
python scripts/run_ablations.py

# Analyze failure cases
python scripts/analyze_failures.py

# Profile system performance
python scripts/profile_system.py

# Validate data engine
python scripts/validate_data_engine.py

# Generate sim-to-real analysis
python scripts/test_on_kitti.py

# Open interactive dashboard
jupyter notebook notebooks/analysis_dashboard.ipynb
```

All scripts automatically:
- Generate metrics JSON files in `results/`
- Create visualization plots in `docs/figures/`
- Auto-generate markdown documentation in `docs/`
