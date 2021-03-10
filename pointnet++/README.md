
# Implement pointnet++ on industry data

## prerequisites
- python 3.8
- pytorch 1.7.0
- cuda 11.0 or higher
- sklean
- time
- os
- pandas
- numpy
- csv
- argparse
- tensorboardX

## Data
Can put the customize data in Data folder in the same format, then change the path in train_source.py, line 27 and 48.

## train the model
    python train_source.py  
  
Can train hyperparameters of training the train_source.py

## output
3 output files will come out:  
classification_report.csv: the classfication report  
test.h5: the trained model  

