# HAR-EFNet: Human Activity Recognition with ECDF Feature Network

This project implements a Human Activity Recognition (HAR) system using ECDF-based feature prediction for representation learning. The system is designed to recognize 12 distinct human activities from IMU sensor data using a two-stage training pipeline:

**1. Encoder Training (Supervised Regression)**
   
A neural encoder is trained to predict ECDF (Empirical Cumulative Distribution Function) features derived from raw 9-channel IMU time-series data.
The model learns to regress a structured [3×78] feature vector (78 features per axis) using MSE loss.

**2. Activity Classification**
   
After training, the encoder is frozen and used as a feature extractor. A lightweight classifier is then trained on top of the predicted ECDF features to perform multi-class classification across 12 activity classes.


---


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
│   ├── deepconvlstm_attn_encoder.py
│   ├── deepconvlstm_encoder.py
│   └── sa_har_encoder.py   
├── classifiers/  
│   ├── __init__.py      
│   ├── deepconvlstm_attn_classifier.py   
│   ├── deepconvlstm_classifier.py           
│   └── sa_har_classifier.py  
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

---


## Installation
Install the dependencies using Poetry:
```bash
poetry install
```
## Usage
### 1. End-to-End Training - All Subjects
```bash
python main.py --train_encoder True --train_classifier True --test True
```

### 2. End-to-End Training - Specific Subject
```bash
python main.py --train_encoder True --train_classifier True --test True --specific_subject 1
``` 
