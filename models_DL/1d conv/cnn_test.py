import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from custom_data import bpdata_train,bpdata_test
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
            nn.Conv2d(1, 16,kernel_size=1, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2))
        self.fc = nn.Linear(32384, 1)
    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        return out

for j in range(20):
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
	    for i,(data,label) in enumerate(test_loader):
	        label = label.float()
	        #print(label)
	        data = torch.tensor(data).float()
	        data = data.unsqueeze(0)
	        #print(data.shape)
	        outputs = model(data)
	        outputs = outputs[0]
	        #print(outputs,label)

	        outputs = outputs.numpy()
	        label = label.numpy()

	        output_list.append(outputs)
	        label_list.append(label)

	        #print(outputs,label)



	    print('Testing MAE in epoch {}: {} '.format(j,metrics.mean_absolute_error(label_list, output_list)))

