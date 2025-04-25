### Attribution

This project incorporates code from the following sources:

- [ISWC22-HAR](https://github.com/teco-kit/ISWC22-HAR), developed by Zhou et al., as described in their paper:  
  Zhou, Y.; Zhao, H.; Huang, Y.; Hefenbrock, M.; Riedel, T.; Beigl, M. (2022). *TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition*. International Symposium on Wearable Computers (ISWC’22). DOI: [10.1145/3544794.3558467](https://doi.org/10.1145/3544794.3558467)

- [VNN](https://github.com/FlyingGiraffe/vnn), developed by Lee et al., as described in their paper:  
  Lee, J., Park, J., Kim, Y., Kim, H. (2021). *VNN: Virtual Node Neural Network for Graph Classification*. Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM ‘21). DOI: [10.1145/3459637.3482383](https://doi.org/10.1145/3459637.3482383)

```
HAR_SSL/
├── configs/                
│   ├── config.py           
│   └── model.yaml          
├── dataloaders/            
│   ├── data_loader.py      
│   └── data_utils.py       
├── encoders/              
│   ├── __init__.py         
│   ├── cnn_encoder.py      
│   └── lstm_encoder.py    
├── classifiers/            
│   └── classifier_base.py  
├── utils/                  
│   ├── __init__.py         
│   ├── training_utils.py   
│   └── logger.py           
├── main.py                 
├── train_encoder.py        
└── train_classifier.py    
```

### The dependencies can be installed by:
```bash
poetry install
```

### 1. Train Encoder
```bash
   python HAR_SSL/main.py -d pamap2 -e cnn --train_encoder True --train_classifier False --test False
   ```

### 2. Train Evaluator
```bash
   python HAR_SSL/main.py -d pamap2 -e cnn --train_encoder False --train_classifier True --test False --load_encoder True --encoder_path /path/to/encoder.pth
   ```

### 3. End-to-End (all subjects)
```bash
   python HAR_SSL/main.py -d pamap2 -e cnn --train_encoder True --train_classifier True --test True
   ```
### 4. End-to-End (specific subject)
```bash
   python HAR_SSL/main.py --train_encoder=True --train_classifier=True --test=True --specific_subject=5
   ```