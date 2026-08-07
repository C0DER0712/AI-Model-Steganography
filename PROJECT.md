Project Name:
Adaptive Deep Learning-based AI Model Steganography for Evaluating AI Model Malware Detectors

Objective

This project is purely for defensive AI security research.

The embedding network is trained to simultaneously:

1. Preserve host model accuracy.

2. Allow the decoder to recover a benign random payload.

3. Produce modified weights that remain difficult for Model X-Ray to detect.

No malware is ever generated or executed.

Random payloads are used exclusively.

Architecture

Host Model
↓

Weight Extraction
↓

Model X-Ray Image Representation

↓

Encoder
↓

Modified Weights

↓

Decoder

↓

Recovered Payload

↓

Frozen Model X-Ray Detector

Training Objectives

Loss =
α Classification Loss
+ β Payload Recovery Loss
+ γ Weight Distortion Loss
+ δ Detector Loss

Evaluation

Accuracy Drop

Payload Recovery

Bit Error Rate

Model X-Ray Detection Rate

Embedding Capacity

Supported Host Models

ResNet18

ResNet50

MobileNetV2

VGG16

Coding Standards

Use PyTorch

Type hints

Google docstrings

Unit tests

No hardcoded paths

Modular code

Publication-quality figures