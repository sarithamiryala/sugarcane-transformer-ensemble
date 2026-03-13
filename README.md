sugarcane-transformer-ensemble
Sugarcane Leaf Disease Classification using Vision Transformers and Ensemble Learning

This repository contains the implementation of the research work:

An Ensemble of Vision and Swin Transformers with Deep CNN Models for Sugarcane Disease Diagnosis

The project focuses on automated classification of sugarcane leaf diseases using deep learning models including:

- Vision Transformer (ViT)
- Swin Transformer
- ResNet50
- DenseNet121
- EfficientNetB0
- MobileNetV2

An ensemble model combining ViT and Swin Transformer is also implemented to improve classification performance.



Dataset

The dataset used in this work is publicly available on Kaggle:

Dataset Link:

https://www.kaggle.com/datasets/akilesh253/sugarcane-plant-diseases-dataset

The dataset contains images of sugarcane leaves belonging to the following classes:

1. Healthy
2. Brown Rust
3. Banded Chlorosis
4. Pokkah Boeng
5. Yellow Leaf Disease
6. Red Rot

Images were resized to 224 × 224 before training.



Repository Structure

 Dataset

The dataset used in this research is publicly available on Kaggle:

https://www.kaggle.com/datasets/akilesh253/sugarcane-plant-diseases-dataset

The dataset contains sugarcane leaf images across six classes:

1. Bacterial Blight
2. Mosaic Disease
3. Red Rot
4. Rust
5. Yellow Leaf
6. Healthy Leaves   


git clone https://github.com/sarithamiryala/sugarcane-transformer-ensemble.git

cd sugarcane-transformer-ensemble

pip install -r requirements.txt