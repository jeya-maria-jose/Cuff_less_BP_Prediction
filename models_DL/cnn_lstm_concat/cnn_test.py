import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from custom_data2 import bpdata_train,bpdata_test
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics  
batch_size =1
bp_test_dataset = bpdata_test(csv_file='/home/jeyamariajose/Projects/dl/bp_test_new.csv',
                                    root_dir='/home/jeyamariajose/Projects/dl/data/cleaned/test/')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = torch.utils.data.DataLoader(dataset=bp_test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

class ConvNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32,kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.lstm = nn.LSTM(32256,100,1)
        self.fc = nn.Linear(32356, 32)
        self.fc2 = nn.Linear(32, 3)
        
    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)


        
        out1 = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = out1.unsqueeze(0)
        #print(out1.shape)
        out, hid = self.lstm(out)

        
        #print(out.shape)
        out = out.squeeze(0)


        out = np.concatenate((out.detach().numpy(),out1.detach().numpy()),axis=None)
        #print(out.shape)
        out = torch.from_numpy(out)
        out = out.unsqueeze(0)

        

        out = self.fc(out)
        out = self.fc2(out)
        #print(out,label)
        
        #print(out.shape)
        return out


sbp=0
mbp=0
dbp=0

for j in range(16,17):
    file_path = "model%d.ckpt"%j
    #file_path = "model.ckpt"
    model = ConvNet(1).to(device)
    model.load_state_dict(torch.load(file_path))

    output_list = list()
    label_list = list()

    model.eval()  
    with torch.no_grad():
        correct = 0
        total = 0
        output_list_sbp = list()
        label_list_sbp = list() 
        output_list_dbp = list()
        label_list_dbp = list()  
        output_list_mbp = list()
        label_list_mbp = list()     
        for i,(data,label) in enumerate(test_loader):
            #label = label.float()
            #print(data,label)

            data = torch.tensor(data).float()
            data = data.unsqueeze(0)
            #print(data)
            #print(data.shape)
            outputs = model(data)
            outputs = outputs[0]
            #printoutputs,label)

            outputs = outputs.detach().numpy()
            #label = label.numpy()

            output_list_sbp.append(outputs[0])
            label_list_sbp.append(label[0].numpy()[0])
            output_list_dbp.append(outputs[1])
            label_list_dbp.append(label[1].numpy()[0])
            output_list_mbp.append(outputs[2])
            label_list_mbp.append(label[2].numpy()[0])

        print('Testing SBP MAE in epoch {}: {} '.format(j,metrics.mean_absolute_error(label_list_sbp, output_list_sbp)))
        print('Testing DBP MAE in epoch {}: {} '.format(j,metrics.mean_absolute_error(label_list_dbp, output_list_dbp)))
        print('Testing MBP MAE in epoch {}: {} '.format(j,metrics.mean_absolute_error(label_list_mbp, output_list_mbp)))

            #print(label_list_sbp)
           
        for gg in range(len(output_list_dbp)):
            print("SBP",output_list_sbp[gg],label_list_sbp[gg])
            print("DBP",output_list_dbp[gg],label_list_dbp[gg])
            print("MBP",output_list_mbp[gg],label_list_mbp[gg])


      #  print(len(output_list_dbp))

#        print("SBP",sbp/501)
 #       print("DBP",dbp/501)
  #      print("MBP",mbp/501)


