# Generalization Analysis: Sim-to-Real Transfer

*Analysis of model performance when transferring from CARLA (synthetic) to real-world data*

## Motivation

Models trained on CARLA (synthetic simulator) may not generalize well to real-world data 
due to domain shift. This analysis documents expected challenges and mitigation strategies.

## Expected Distribution Shift

### CARLA (Training Domain)
- **Lighting**: Perfect, consistent lighting with no glare or shadows
- **Textures**: Clean, high-quality textures without weathering
- **Sensor**: No sensor noise, motion blur, or compression artifacts
- **Weather**: Controlled conditions, limited weather variations
- **Objects**: Limited object types and variations

### Real-World (Target Domain)
- **Lighting**: Highly variable with glare, shadows, and varying sun positions
- **Textures**: Worn roads, faded lane markings, debris, and weathering
- **Sensor**: Camera noise, motion blur, lens distortion, compression artifacts
- **Weather**: Rain, fog, snow, and varying atmospheric conditions
- **Objects**: Diverse vehicle types, pedestrians, cyclists, and unexpected objects

## Expected Performance Degradation

| Metric | CARLA (Sim) | Real-World Expected | Performance Drop |
|--------|-------------|---------------------|------------------|
| Detection mAP@0.5 | 0.78 | 0.52-0.62 | 20-33% |
| Lane IoU | 0.85 | 0.58-0.68 | 20-32% |
| Drivable IoU | 0.88 | 0.65-0.75 | 15-26% |
| Policy RMSE | 0.09 | 0.18-0.25 | 100-178% increase |

## Key Challenges

### 1. Visual Appearance Gap
- **Issue**: Synthetic textures look different from real photos
- **Impact**: Detection precision drops, segmentation boundaries less accurate
- **Example**: Perfect lane markings in sim vs faded/damaged in reality

### 2. Lighting and Weather Variation
- **Issue**: Limited lighting/weather diversity in training
- **Impact**: Failure in challenging conditions (glare, fog, night)
- **Example**: Models fail in overcast conditions not seen during training

### 3. Sensor Characteristics
- **Issue**: Real cameras have noise, blur, and artifacts
- **Impact**: Reduced feature quality, missed detections
- **Example**: Motion blur causes segmentation failures

### 4. Behavioral Distribution Shift
- **Issue**: Real-world driving patterns differ from sim
- **Impact**: Policy predictions don't generalize
- **Example**: Aggressive drivers, jaywalking pedestrians not in sim

## Mitigation Strategies

### 1. Domain Randomization
**Approach**: Randomize sim parameters during training

- Lighting: Random sun position, intensity, shadows
- Weather: Random fog, rain, cloud cover
- Textures: Random wear, fading, weathering
- Camera: Add synthetic noise, blur, lens distortion

**Expected Improvement**: 5-10% recovery in real-world performance

### 2. Style Transfer (CycleGAN)
**Approach**: Learn mapping from sim images to realistic images

- Train CycleGAN on unpaired sim/real images
- Apply style transfer during training or inference
- Preserve semantic content while changing appearance

**Expected Improvement**: 8-15% recovery in real-world performance

### 3. Fine-Tuning on Real Data
**Approach**: Adapt pre-trained model on small real-world dataset

- Collect 1,000-5,000 real-world labeled frames
- Fine-tune last layers while freezing early layers
- Use mixed batches (80% sim, 20% real)

**Expected Improvement**: 15-25% recovery in real-world performance

### 4. Domain Adaptation (Adversarial)
**Approach**: Train domain discriminator to align feature distributions

- Add domain classifier to distinguish sim vs real features
- Adversarially train to make features domain-invariant
- Preserve task performance while reducing domain gap

**Expected Improvement**: 10-18% recovery in real-world performance

### 5. Sim-to-Real Data Mixing
**Approach**: Combine synthetic and real data during training

- Use sim data for base training (abundant, cheap)
- Augment with small real dataset (expensive, scarce)
- Weight real examples higher in loss function

**Expected Improvement**: 12-20% recovery in real-world performance

## Recommended Pipeline

1. **Phase 1**: Train on CARLA with heavy domain randomization
2. **Phase 2**: Apply style transfer or collect small real dataset
3. **Phase 3**: Fine-tune on real data with mixed batches
4. **Phase 4**: Validate on diverse real-world scenarios
5. **Phase 5**: Iteratively collect failure cases and retrain

## Expected Final Performance

With full mitigation pipeline:

| Metric | CARLA Baseline | Real-World (No Adaptation) | Real-World (With Adaptation) |
|--------|----------------|----------------------------|------------------------------|
| Detection mAP@0.5 | 0.78 | 0.52 | 0.68-0.72 |
| Lane IoU | 0.85 | 0.58 | 0.72-0.78 |
| Policy RMSE | 0.09 | 0.25 | 0.13-0.16 |

## Testing on KITTI (Future Work)

To validate these hypotheses, testing on KITTI dataset would involve:

1. Download KITTI raw driving sequences
2. Run inference with CARLA-trained models
3. Manually annotate subset for quantitative evaluation
4. Analyze failure modes specific to real-world data
5. Implement and evaluate mitigation strategies

## References

- CARLA Simulator: Dosovitskiy et al., 2017
- KITTI Dataset: Geiger et al., 2012
- Domain Randomization: Tobin et al., 2017
- CycleGAN: Zhu et al., 2017
