import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from custom_data import bpdata_train,bpdata_test
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics  

#Hyper-parameters

batch_size =1
num_classes = 7

bp_test_dataset = bpdata_train(csv_file='/home/jeyamariajose/Projects/dl/bp_train_new.csv',
                                    root_dir='/home/jeyamariajose/Projects/dl/data/cleaned/train/')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = torch.utils.data.DataLoader(dataset=bp_test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

class ConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16,kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.fc1 = nn.Linear(64000, 32)
        self.fc2 = nn.Linear(32, 7)
        #self.sm = nn.Softmax()

        
    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        #out = self.sm(out)
        return out

for j in range(1,20):
    file_path = "model%d.ckpt"%j
    #file_path = "model.ckpt"
    model = ConvNet(num_classes).to(device)
    model.load_state_dict(torch.load(file_path))

    output_list = list()
    label_list = list()

    model.eval()  
    with torch.no_grad():
        correct = 0
        total = 0
        for  (data, mbp_data, labels )in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            data = torch.tensor(data).float()
            data = data.unsqueeze(0)
            outputs = model(data)

            
            _, predicted = torch.max(outputs.data, 1)

            #print("predicted",predicted)
            #print("label",labels)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #print(correct,total)

        print('Test Accuracy of the model on the 10000 test data: {} %'.format(100 * correct / total))
