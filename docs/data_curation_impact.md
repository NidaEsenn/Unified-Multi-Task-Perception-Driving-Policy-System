# Data Curation Impact Analysis

*Auto-generated report on data engine effectiveness*

## Experiment Setup

- **Raw Dataset**: 10000 frames
- **Curated Dataset**: 7234 frames
- **Frames Removed**: 2766 (27.7%)

## Data Distribution Analysis

### Before Curation

- Mean steering: 0.020
- Std deviation: 0.450
- Blur percentage: 18.5%
- Overexposed: 12.3%

### After Curation

- Mean steering: 0.010
- Std deviation: 0.520 (↑ 16% diversity)
- Blur percentage: 3.2% (↓ 83% improvement)
- Overexposed: 2.1% (↓ 83% improvement)

## Curation Strategy Impact

| Filter | Frames Removed | Expected Impact |
|--------|----------------|-----------------|
| Blur detection | 1850 (18.5%) | +15% sharpness improvement |
| Brightness filter | 615 (6.2%) | +8% balanced exposure |
| Motion magnitude | 301 (3.0%) | +5% diversity in scenarios |

## Expected Performance Improvements

- **detection_map**: +3-5% expected improvement
- **segmentation_iou**: +4-6% expected improvement
- **policy_rmse**: -8-12% expected reduction
- **training_efficiency**: 30% faster convergence expected

## Key Findings

1. **Quality Improvement**: 83% reduction in low-quality frames (blur, poor exposure)
2. **Distribution Balance**: 16% increase in steering diversity after removing straight-road bias
3. **Training Efficiency**: Expected 30% faster convergence due to higher-quality training data
4. **Performance Impact**: Expected 4-6% improvement in segmentation, 3-5% in detection

## Recommendations

- Continue using blur and brightness filtering
- Consider adding semantic filtering (e.g., remove frames with no vehicles)
- Implement active learning to identify edge cases
- Balance steering distribution further with targeted sampling
