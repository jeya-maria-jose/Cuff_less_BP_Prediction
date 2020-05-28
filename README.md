# Cuff less Blood Pressure Prediction 

This repository hosts the code for Prediction of Blood Pressure from ECG and PPG signals using two methods.

1. Feature Extraction and Regression using Machine Learning Methods. <a href="https://www.sciencedirect.com/science/article/abs/pii/S1746809420300987"> Paper </a>

2. Deep learning based regression.



### Getting Started:

- Clone this repo:
```bash
git clone https://github.com/jeya-maria-jose/Cuff_less_BP_Prediction
cd Cuff_less_BP_Prediction
```

### Dataset:

Dataset :  [Link](https://archive.ics.uci.edu/ml/machine-learning-databases/00340/)

This database consist of a cell array of matrices, each cell is one record part. 

In each matrix each row corresponds to one signal channel: 

1: PPG signal, FS=125Hz; photoplethysmograph from fingertip 

2: ABP signal, FS=125Hz; invasive arterial blood pressure (mmHg) 

3: ECG signal, FS=125Hz; electrocardiogram from channel II 


## Feature Extraction and Machine Learning based method:

### Prerequisites:

- MATLAB
- Python 3
- Scikit-learn

### Feature Extraction

The features taken are explained <a href="https://sites.google.com/view/cufflessbp/features-notes">here </a>

<code> seven_features.m </code> - Code to extract the features : (WN,PIR,PTT,HR,IH,IL,Meu)

<code> ppg_features.m </code> - Code to extract the PPG features 

<code> PTT_final.m </code> - Code to extract the PTT 

The extracted features are saved in a CSV file from MATLAB.

The CSV file : <a href = "https://drive.google.com/file/d/19mflxMXKuGKNLUM8Uirgg1P0JeguRs7e/view?usp=sharing"> Link </a>
The columns denote the features and BP GT in the same order as extracted.

### Machine Learning models

```bash
cd models_ML
python rf.py
```


## Using the DL Code:

### Prerequisites:

- Linux 
- Python 3 
- Pytorch

### Training

```bash
cd models_DL/cnn_lstm_concat
python cnn_multitask.py
```

### Testing 


```bash
cd models_DL/cnn_lstm_concat
python cnn_test.py
```
### Disclaimer

The code is not completely clean as the data directories are initialized manually. Please make sure the directories are changed according to the remote server where the code is run. 

## Citation
If you use this , please cite our paper <a href="https://www.sciencedirect.com/science/article/abs/pii/S1746809420300987"> Investigation on the effect of Womersley number, ECG and PPG features for cuff less blood pressure estimation using machine learning</a>:

### ML Experiments and Womersley number Paper - 

```
@article{thambiraj2020investigation,
  title={Investigation on the effect of Womersley number, ECG and PPG features for cuff less blood pressure estimation using machine learning},
  author={Thambiraj, Geerthy and Gandhi, Uma and Mangalanathan, Umapathy and Jose, V Jeya Maria and Anand, M},
  journal={Biomedical Signal Processing and Control},
  volume={60},
  pages={101942},
  year={2020},
  publisher={Elsevier}
}
}
```
## Results for DL Experiments - Coming Soon

This work was done while at National Institute of Technology, Tiruchirapalli; India

