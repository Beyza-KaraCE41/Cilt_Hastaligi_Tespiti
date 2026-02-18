# Skin Disease Classification Project

## 1. Project Purpose and Importance

Automatic classification of skin diseases plays an important role in medical diagnosis systems. This project aims to develop deep learning models using images of **5 different skin diseases** (Acne, Hyperpigmentation, Nail Psoriasis, SJS-TEN, Vitiligo).

Accurate diagnosis of diseases is critical, which is why a high-accuracy model is essential. In this project, **24 different models** were trained to identify the best-performing one.

### Main Goals

- Perform Exploratory Data Analysis (EDA) on the dataset
- Develop 24 deep learning / transfer learning models
- Build an original weighted ensemble model
- Analyze all models via Accuracy, Loss, Confusion Matrix, and ROC curves
- Prevent overfitting
- Perform parameter optimization
- Save the best model and make it ready for predictions on new samples

---

## 2. Dataset and Exploratory Data Analysis (EDA)

The dataset consists of **5 classes**. Before training, data was normalized and augmentation techniques were applied.

### Classes and Characteristics

| Class | Description |
|-------|-------------|
| **Acne** | Young skin, inflamed lesions |
| **Hyperpigmentation** | Color change, dark spots |
| **Nail Psoriasis** | Nail deformations, pitting |
| **SJS-TEN** | Severe skin reaction, orange-red |
| **Vitiligo** | Pigment loss, white patches |

![Class Distribution](image1.jpg)

![Sample Images](image2.jpg)

![EDA Visualization](image3.jpg)

> **Note:** An imbalanced distribution was observed across classes (SJS-TEN had the most samples). This was balanced during training using the `class_weight` parameter.

### EDA Findings

- Balanced distribution across classes
- High image quality
- Different skin tones represented
- Variable lighting conditions

### Preprocessing

**Normalization:** Pixel values rescaled to 0-1 range

**Data Augmentation:**
- Rotation: 30 degrees
- Shift: 30%
- Zoom: 0.8x - 1.2x
- Flip: Horizontal
- Brightness: 0.85 - 1.15

---

## 3. Developed Hybrid Model Architecture

To combine the feature extraction capabilities of individual models, **DenseNet121, InceptionV3, and ResNet50V2** were connected in parallel. Features from these models were merged in a Concatenate layer, then passed through customized Dense layers (512, 256, 128 neurons).

![Hybrid Model Architecture](image4.png)

---

## 4. Methods Used

### Base Transfer Learning Models

| Model | Parameters | Accuracy | Notes |
|-------|-----------|----------|-------|
| **DenseNet121** | 7.04M | 87.53% | Dense connections; last 20 layers fine-tuned, L2 regularization |
| **InceptionV3** | 21.8M | 86.24% | Multi-scale convolutions (1x1, 3x3, 5x5); Dropout(0.3) + BatchNorm |
| **ResNet50V2** | 23.6M | 83.26% | Residual connections; Early stopping + ReduceLROnPlateau |
| **EfficientNetB0** | 5.3M | 27.75% | Compound scaling - incompatible with this dataset |
| **VGG16** | 138M | 11.65% | Classic sequential CNN - outdated architecture, very poor performance |
| **Xception** | 22.9M | 83.08% | Depthwise separable convolutions |

### Optimized Models

| Model | Accuracy | Key Feature |
|-------|----------|-------------|
| **Optimized_Model** | 96.60% | Ensemble + Fine-tuning; L2(0.0002) + Dropout(0.25) + BatchNorm + Early Stop |
| **Premium_v4_Model** | 87.35% | Ensemble of 4 networks, fine-tuned |
| **Final_50Epoch_Model** | 96.08% | Long training (50 epochs); Cascading dropout (0.3 to 0.2) |

---

## 5. Champion Model - Custom Hybrid Weighted Ensemble

### Custom_Hybrid_Weighted.h5 - CHAMPION

The secret of success: Weighted ensemble concept

| Property | Value |
|----------|-------|
| **Architecture** | Weighted ensemble |
| **Accuracy** | 98.07% |
| **Optimizer** | AdamW (lr=1e-4) |
| **Regularization** | L2(0.0002) + Dropout(0.3 to 0.2) + BatchNorm |

### Components

| Model | Weight |
|-------|--------|
| DenseNet121 | 30% |
| InceptionV3 | 25% |
| ResNet50V2 | 25% |
| EfficientNetB3 | 20% |

### Why 98.07% is so successful?

- Standard concatenation (86%) vs Weighted (98%) = +12% improvement
- DenseNet's strong features prioritized (30%)
- Weak networks' impact minimized (20%)
- Optimal weight combination

### Overfitting Prevention Methods

| Method | Details |
|--------|---------|
| **L2 Regularization (0.0002)** | Penalizes weights to reduce model complexity |
| **Dropout Cascading (0.3 to 0.2)** | Gradual neuron deactivation per layer |
| **BatchNormalization** | Activation normalization on every dense layer |
| **AdamW Optimizer** | Weight decay reduces overfitting tendency |
| **Early Stopping (patience=10)** | Stops when validation accuracy starts to drop |
| **ReduceLROnPlateau** | Reduces learning rate when val loss plateaus |

---

## 6. Model Comparison

### Full Performance Table

![All Models Performance Table](image5.png)

---

## 7. Detailed Model Results

### Champion - Custom_Hybrid_Weighted (98.07%)

![Confusion Matrix - Champion](image6.jpg)

**Confusion Matrix:** Diagonal values range from 97-99%. Vitiligo and SJS-TEN are recognized with 99% accuracy, while Nail Psoriasis reaches 97%.

![ROC Curve - Champion](image7.jpg)

**ROC Curve:** AUC 0.98+ for all classes. Micro-average AUC: 0.98. ROC curves are very close to the top-left corner, representing ideal performance.

![Accuracy-Loss - Champion](image8.png)

**Accuracy-Loss Graphs:** Training and validation accuracy are nearly identical (both 98%). Gap: 0.07% - Excellent! This proves the model generalizes perfectly without overfitting.

---

### Final_50Epoch_Model (96.08%)

![Confusion Matrix - 50 Epoch](image9.jpg)

**Confusion Matrix:** 95-97% diagonal accuracy. Vitiligo is the strongest class (97%).

![ROC Curve - 50 Epoch](image10.jpg)

**ROC Curve:** AUC 0.95+ for all classes.

![Accuracy-Loss - 50 Epoch](image11.jpg)

**Accuracy-Loss Graphs:** Training: 96%, Validation: 96% (Gap: 0%). No overfitting thanks to cascading dropout and early stopping.

---

### Optimized_Model (96.60%)

![Confusion Matrix - Optimized](image12.jpg)

**Confusion Matrix:** 90-94% diagonal. Hyperpigmentation is the weakest class (88%).

![ROC Curve - Optimized](image13.jpg)

**ROC Curve:** AUC 0.90-0.92.

![Accuracy-Loss - Optimized](image14.png)

**Accuracy-Loss Graphs:** Training: 93%, Validation: 92% (Gap: 1%).

---

### Premium_v4_Model (87.35%)

![Confusion Matrix - Premium](image15.jpg)

**Confusion Matrix:** 86-88% diagonal. Acne and Hyperpigmentation confused.

![ROC Curve - Premium](image16.jpg)

**ROC Curve:** AUC 0.86-0.88.

![Accuracy-Loss - Premium](image17.jpg)

**Accuracy-Loss Graphs:** Training: 87%, Validation: 87.35% (Gap: 0.35%).

---

### DenseNet121 (87.53%)

![Confusion Matrix - DenseNet121](image18.jpg)

**Confusion Matrix:** 85-89% diagonal. Class confusions at 11-15%.

![ROC Curve - DenseNet121](image19.jpg)

**ROC Curve:** AUC 0.87-0.89.

![Accuracy-Loss - DenseNet121](image20.jpg)

**Accuracy-Loss Graphs:** Training: 88%, Validation: 87.5% (Gap: 0.5%).

---

### InceptionV3 (86.24%)

![Confusion Matrix - InceptionV3](image21.jpg)

**Confusion Matrix:** 84-88% diagonal. 14-16% error rate.

![ROC Curve - InceptionV3](image22.jpg)

**ROC Curve:** AUC 0.85-0.87.

![Accuracy-Loss - InceptionV3](image23.jpg)

**Accuracy-Loss Graphs:** Training: 86.5%, Validation: 86.24% (Gap: 0.26%).

---

### Xception (83.08%)

![Confusion Matrix - Xception](image24.jpg)

**Confusion Matrix:** 81-85% diagonal.

![ROC Curve - Xception](image25.jpg)

**ROC Curve:** AUC 0.82-0.84.

![Accuracy-Loss - Xception](image26.jpg)

**Accuracy-Loss Graphs:** Training: 83.3%, Validation: 83.08% (Gap: 0.22%).

---

### ResNet50V2 (83.26%)

![Confusion Matrix - ResNet50V2](image27.jpg)

**Confusion Matrix:** 81-85% diagonal. 15-19% error.

![ROC Curve - ResNet50V2](image28.jpg)

**ROC Curve:** AUC 0.82-0.84.

![Accuracy-Loss - ResNet50V2](image29.jpg)

**Accuracy-Loss Graphs:** Training: 83.5%, Validation: 83.26% (Gap: 0.24%).

---

### EfficientNetB0 (29.78%) - Failed

![Confusion Matrix - EfficientNetB0](image30.jpg)

**Confusion Matrix:** 27.76% diagonal. Model predicts all samples as SJS-TEN. Error rate: 72.24%.

![ROC Curve - EfficientNetB0](image31.jpg)

**ROC Curve:** Average AUC approximately 0.61.

![Accuracy-Loss - EfficientNetB0](image32.jpg)

**Accuracy-Loss Graphs:** Underfitting problem - strong SJS-TEN bias.

---

### VGG16 (16.42%) - Failed

![Confusion Matrix - VGG16](image33.jpg)

**Confusion Matrix:** 11.37% diagonal. Error rate: 88.63%.

![ROC Curve - VGG16](image34.jpg)

**ROC Curve:** Average AUC approximately 0.50 - random guess level.

![Accuracy-Loss - VGG16](image35.jpg)

**Accuracy-Loss Graphs:** Underfitting problem - outdated architecture.

---

## 8. Overfitting Prevention Strategy Summary

| Method | Value | Purpose |
|--------|-------|---------|
| **L2 Regularization** | 0.0002-0.0005 | Penalize weights to reduce complexity |
| **Dropout Cascading** | 0.3 to 0.25 to 0.2 | Gradually deactivate neurons per layer |
| **BatchNormalization** | Every dense layer | Normalize activations |
| **Early Stopping** | patience=8-10 | Stop when validation accuracy drops |
| **ReduceLROnPlateau** | factor=0.6-0.7, patience=3-5 | Reduce learning rate at plateaus |
| **Data Augmentation** | Rotation, Shift, Zoom, Flip, Brightness | Expose model to more variations |
| **Fine-tuning** | Base frozen, last layers trainable | Keep ImageNet features |

---

## 9. Sample Predictions with the Champion Model

Each disease category had one sample selected. The model made predictions on these samples, demonstrating how well it applies what it learned to real-world data.

![Prediction Sample 1](image36.png)
![Prediction Result 1](image37.png)

![Prediction Sample 2](image38.png)
![Prediction Result 2](image39.png)

![Prediction Sample 3](image40.png)
![Prediction Result 3](image41.png)

![Prediction Sample 4](image42.png)

![Prediction Sample 5](image43.png)

---

## Summary

| Model | Accuracy | Type |
|-------|----------|------|
| **Custom_Hybrid_Weighted** | **98.07%** | Original Weighted Ensemble |
| Optimized_Model | 96.60% | Ensemble + Fine-tuning |
| Final_50Epoch_Model | 96.08% | Long Training Ensemble |
| Premium_v4_Model | 87.35% | Standard Ensemble |
| DenseNet121 | 87.53% | Transfer Learning |
| InceptionV3 | 86.24% | Transfer Learning |
| ResNet50V2 | 83.26% | Transfer Learning |
| Xception | 83.08% | Transfer Learning |
| EfficientNetB0 | 29.78% | Failed |
| VGG16 | 16.42% | Failed |
