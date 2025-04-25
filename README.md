# HAR-EFNet: Human Activity Recognition with ECDF Feature Network

## Introduction

This project implements a Human Activity Recognition (HAR) system using ECDF-based feature prediction for representation learning. The system is designed to recognize 12 distinct human activities from IMU sensor data using a two-stage training pipeline:

1.	Encoder Training (Supervised Regression)
A neural encoder is trained to predict 234-dimensional ECDF (Empirical Cumulative Distribution Function) features from raw IMU signals using MSE loss.

2.	Activity Classification
After training, the encoder is used as a fixed feature extractor. A separate classifier is then trained on top of the predicted ECDF features to recognize 12 distinct human activities.

## Project Structure
```
HAR-EFNet/
├── configs/                
│   ├── config.py           
│   ├── model.yaml          
│   └── data.yaml
├── dataloaders/            
│   ├── data_loader.py      
│   └── data_utils.py       
├── encoders/              
│   ├── __init__.py         
│   ├── base.py
│   ├── cnn_encoder.py
│   └── lstm_encoder.py   
├── classifiers/  
│   ├── __init__.py           
│   └── classifier_base.py  
├── utils/                  
│   ├── __init__.py         
│   ├── training_utils.py   
│   └── logger.py           
├── main.py                 
├── train_encoder.py        
└── train_classifier.py
```

Additional directories created during execution:
```
HAR-EFNet/
├── logs/
│   ├── debug/ 
│   └── training/
└── saved/
    ├── encoders/
    ├── classifiers/
    └── results/
```

### The dependencies can be installed by:
```bash
poetry install
```

### 1. Train Encoder
```bash
python main.py -d pamap2 -e cnn --train_encoder True --train_classifier False --test False
```

### 2. Train Classifier with Pre-trained Encoder
```bash
python main.py -d pamap2 -e cnn --train_encoder False --train_classifier True --test False --load_encoder True --encoder_path /path/to/encoder.pth
```

### 3. End-to-End Training and Testing (all subjects)
```bash
python main.py -d pamap2 -e cnn --train_encoder True --train_classifier True --test True
```

### 4. End-to-End for Specific Subject
```bash
python main.py -d pamap2 -e cnn --train_encoder True --train_classifier True --test True --specific_subject 5
``` 