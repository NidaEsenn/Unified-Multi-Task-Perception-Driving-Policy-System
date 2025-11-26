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




