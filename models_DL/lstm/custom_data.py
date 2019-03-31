from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class bpdata_train(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        
        self.bp = pd.read_csv(csv_file, names = ['sbp','dbp','mbp','cla'])
        self.root_dir = root_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):

        return len(self.bp)

    def __getitem__(self, idx):

        file_name = os.path.join(self.root_dir,'check%d.csv'%(idx+1))
        patient_file = pd.read_csv(file_name,names = ['ppg','ecg'])
        
        #patient_file = patient_file[['ppg','ecg']]
        # patient_file=np.asarray(patient_file)
        patient_file = np.asarray(patient_file)
        # print(img_as_np.shape)
        #patient_file = self.to_tensor(img_as_np)

        #print(patient_file.shape)

        #print(self.bp)
        sbp_list = self.bp['sbp']
        dbp_list = self.bp['dbp']
        mbp_list = self.bp['mbp']
        #print(sbp_list[1])
        sbp_data = sbp_list[idx]
        dbp_data = dbp_list[idx]
        mbp_data = mbp_list[idx]
        
        sample = {'file': patient_file, 'sbp': sbp_data,'dbp': dbp_data,'mbp': mbp_data}

        if self.transform:
            sample = self.transform(sample)

        return patient_file,mbp_data

# bp_dataset = bpdata(csv_file='/home/jeyamariajose/Projects/dl/bp.csv',
#                                     root_dir='/home/jeyamariajose/Projects/dl/data/')

# train_loader = torch.utils.data.DataLoader(dataset=bp_dataset,
#                                            batch_size=batch_size, 
#                                            shuffle=True)

# for i,(data,label) in enumerate(train_loader):
#     print(label)
    


class bpdata_test(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        
        self.bp = pd.read_csv(csv_file, names = ['sbp','dbp','mbp','cla'])
        self.root_dir = root_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):

        return len(self.bp)

    def __getitem__(self, idx):

        file_name = os.path.join(self.root_dir,'check%d.csv'%(idx+4011))
        patient_file = pd.read_csv(file_name,names = ['ppg','ecg'])
        
        #patient_file = patient_file[['ppg','ecg']]
        # patient_file=np.asarray(patient_file)
        patient_file = np.asarray(patient_file)
        # print(img_as_np.shape)
        #patient_file = self.to_tensor(img_as_np)

        #print(patient_file.shape)

        #print(self.bp)
        sbp_list = self.bp['sbp']
        dbp_list = self.bp['dbp']
        mbp_list = self.bp['mbp']
        #print(sbp_list[1])
        sbp_data = sbp_list[idx]
        dbp_data = dbp_list[idx]
        mbp_data = mbp_list[idx]
        
        sample = {'file': patient_file, 'sbp': sbp_data,'dbp': dbp_data,'mbp': mbp_data}

        if self.transform:
            sample = self.transform(sample)

        return patient_file,mbp_data