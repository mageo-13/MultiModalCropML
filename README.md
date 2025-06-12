# MultiModalCropML

Multimodal crop classification using time series of Sentinel 2 Optical and Sentinel 1 SAR remote sensing data, leveraging machine learning techniques.

## Overview
Agricultural monitoring often benefits from the fusion of optical and SAR data due to their complementary characteristics. 
Therefore this project integrates optical and radar (SAR) time series from Google Earth Engine to classify agricultural crops.
Machine Learning Algorithms like Random Forest and Balanced Random Forest in case of immabalnced samples across classes are leveraged. 

## 📂 Folder Structure
- `timeseries/`: GEE JavaScript code to extract timesereies features
-  Preprocessing scripts and classification code
- `data/`: Sample exported timesereis data 

## 🚀 Tools & Technologies
- Google Earth Engine (optical + SAR)
- Python (scikit-learn, pandas, numpy)
- Machine Learning 

## 📈 Future Extensions
- Temporal Deep Learning 
- Explainable AI (XAI) for crop models
