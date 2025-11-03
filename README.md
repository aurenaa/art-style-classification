# 🎨 Multimodal Art Style Classification

A deep learning project that classifies artistic painting styles using Convolutional Neural Networks (CNN) and multimodal approach combining images and metadata.

## 📊 Project Overview

This project implements and compares two models for art style classification:
- **CNN Model**: Uses only visual features from images
- **Multimodal Model**: Combines visual features with metadata (artist, year, genre)

## 🚀 Results

| Model | Accuracy | F1-Score (Weighted) |
|-------|----------|---------------------|
| CNN Only | 26.65% | 22.59% |
| CNN + Metadata | **44.01%** | **41.60%** |

**Key Finding**: The multimodal approach shows **65% improvement** in accuracy compared to using images alone.

## 🏗️ Architecture

### CNN Model
- **Input**: 160×160×3 images
- **Feature Extraction**: 4 Conv2D + MaxPooling2D layers
- **Classification**: Flatten → Dense(256) → Dense(50, softmax)

### Multimodal Model
- Combines CNN visual features with metadata embeddings
- Metadata includes: artist, creation year, genre
- Early stopping with patience=7 to prevent overfitting
