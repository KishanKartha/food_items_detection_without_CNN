# 🍽️ Multi-Object Food Detection (Computer Vision)

A computer vision project for detecting and classifying multiple food items from a single plate image using classical image processing and feature engineering techniques.

## 📌 Overview

This project focuses on identifying different food items in an image without using deep learning (CNNs). Instead, it relies on:

- Image segmentation
- Region of Interest (ROI) extraction
- Hand-crafted feature extraction (colour + texture)
- A lightweight neural network classifier

The goal was to explore fundamental computer vision techniques and build a complete pipeline from scratch.

---

## 🚀 Features

- Detects multiple food items in a single image
- Automatically extracts regions of interest (ROIs)
- Classifies food into predefined categories
- Avoids CNNs — uses classical CV + feature engineering

---

## 🧠 Pipeline

### 1. Region of Interest (ROI) Detection

- Graph-based image segmentation is used to group similar regions.
- K-means clustering reduces colour complexity (e.g. ~200 → 10 clusters).
- Masks are generated for each cluster.
- Contours are extracted using OpenCV.
- Bounding boxes are drawn around detected regions.

---

### 2. ROI Extraction

- Each detected region is cropped from the original image.
- Many irrelevant regions are generated initially.
- Filtering is later done via classification confidence.

---

### 3. Feature Extraction

#### Texture Features
- Gabor filters (multi-scale, multi-orientation)
- Mean and standard deviation of filter responses

#### Colour Features
- Circular mask applied to remove background (plate, utensils, etc.)
- Mean and standard deviation of RGB channels

#### Final Feature Vector

---

### 4. Dataset

- Custom dataset with 4 classes:
  - Soup
  - Meat
  - Rice
  - Noodles / Pasta

- Features extracted and stored in a pandas DataFrame

---

### 5. Classification

- A simple neural network is trained on the feature vectors
- Model predicts labels for each ROI
- Low-confidence predictions are discarded

---

## 🧪 Results

- Successfully detects and classifies multiple food items
- Produces bounding boxes + labelled outputs
- Works reasonably well for structured plate images

---

## ⚠️ Limitations

- Many redundant ROIs are generated
- Dataset quality is inconsistent
- Limited number of classes
- Performance depends heavily on lighting and image quality

---

## 🔧 Future Improvements

- Improve ROI extraction to reduce noise
- Use better feature descriptors
- Expand dataset with more diverse food items
- Include Indian food categories
- Explore hybrid approaches (classical + deep learning)

---

## 🛠️ Tech Stack

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn / Neural Network libraries

---

## 📷 Example Workflow

1. Input image of food plate  
2. Segmentation + clustering  
3. ROI detection (bounding boxes)  
4. Cropping regions  
5. Feature extraction  
6. Classification  
7. Final labelled output  

---

## 📚 Motivation

This project was built as part of a computer vision course to gain hands-on experience with:

- Image segmentation
- Feature engineering
- Object detection pipelines (without CNNs)

---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome.

---

## 📜 License

Specify your licence here (MIT, Apache, etc.)


