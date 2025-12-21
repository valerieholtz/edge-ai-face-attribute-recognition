# Edge AI Face Attribute Recognition  
**Age, Gender, and Facial Expression Recognition on iOS**

---

## Project Overview
This project implements an **on-device facial analysis system** that jointly predicts **age group**, **gender**, and **facial expression** from a single image. The focus is on **edge AI deployment** for iOS, emphasizing **low-latency inference, computational efficiency, and privacy-preserving execution**.

The system follows a **multi-task learning** approach using a shared lightweight backbone and task-specific output heads. The complete pipeline covers model training, evaluation, conversion to Core ML, and integration into a SwiftUI-based iOS application.

---

## Model Architecture
- **Backbone:** MobileNetV2 (ImageNet-pretrained)
- **Architecture:** Shared backbone with multiple task-specific heads
- **Prediction heads:**
  - Age classification (3 classes: child, adult, elderly)
  - Gender classification (2 classes: male, female)
  - Facial expression recognition (7 classes)

This design reduces redundant computation and memory usage compared to training separate models and is well suited for deployment on resource-constrained edge devices.

---

## Datasets
The system is trained using two public benchmark datasets, loaded via Hugging Face Datasets:

- **FairFace** (age and gender classification)  
  https://huggingface.co/datasets/HuggingFaceM4/FairFace  

- **FER2013** (facial expression recognition)  
  https://huggingface.co/datasets/AutumnQiu/fer2013  

The original FairFace age categories are merged into three age groups to reduce complexity and residual class imbalance.

---

## Training Strategy
Training is performed in two stages:

1. **Age and Gender Training**  
   - Backbone and age/gender heads are trained jointly on FairFace  
   - Transfer learning from ImageNet-pretrained weights  
   - Early stopping based on validation loss

2. **Emotion Training**  
   - Backbone and existing heads are frozen  
   - Only the emotion head is trained on FER2013  
   - This selective fine-tuning strategy reduces training time and limits catastrophic forgetting

All experiments are conducted in **Google Colab**, using GPU acceleration when available.

---

## Deployment Pipeline (Edge AI)
The trained PyTorch model is deployed to iOS using the following pipeline:

```
PyTorch → TorchScript → Core ML (.mlpackage) → iOS App
```

- The model architecture is reconstructed for deployment and loaded with trained weights
- TorchScript is used to serialize the model into a static, Python-independent format
- Conversion to Core ML is performed using `coremltools`
- Core ML automatically selects the most suitable compute units (CPU, GPU, Neural Engine) at runtime

This pipeline enables fully **on-device inference** without any network dependency.

---

## iOS Application
A minimal **SwiftUI** application integrates the Core ML model and performs inference directly on the device.

Key features:
- Image selection from the photo library
- Explicit image preprocessing (resize to 224 × 224, ImageNet normalization)
- On-device prediction of age, gender, and facial expression
- Benchmark mode measuring inference latency over multiple runs

Inference is executed synchronously using Apple’s Core ML runtime.

---

## On-Device Evaluation
To assess real-world performance, a small evaluation dataset is collected directly on the iOS device:

- **20 images** balanced across:
  - adult / elderly
  - male / female
  - happy / sad facial expressions
- **5 additional baby images** used as out-of-distribution (OOD) samples

Results show a clear performance gap compared to validation metrics, highlighting the impact of **domain shift, dataset bias, and label-space mismatch**. While inference latency is low and stable, predictive performance degrades under real-world conditions, especially for emotion recognition and out-of-distribution inputs.

These findings underline the importance of careful evaluation design and domain alignment when deploying biometric models on edge devices.
