import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from custom_data import bpdata_train,bpdata_test
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics  
from torch.autograd import Variable
bp_train_dataset = bpdata_train(csv_file='/home/jeyamariajose/Projects/dl/bp_train.csv',
                                    root_dir='/home/jeyamariajose/Projects/dl/data/train')

bp_test_dataset = bpdata_test(csv_file='/home/jeyamariajose/Projects/dl/bp_test.csv',
                                    root_dir='/home/jeyamariajose/Projects/dl/data/test/')



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameers
num_epochs = 20
num_classes = 1
batch_size = 1
learning_rate = 0.001


train_loader = torch.utils.data.DataLoader(dataset=bp_train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=bp_test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


class ConvNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8,kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(12096, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        #nn.init.xavier_uniform_(self.layer1.weight)
        #nn.init.xavier_uniform_(self.layer2.weight)
        
    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)
min_loss = 1000
# Loss and optimizer
criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i,(data,label) in enumerate(train_loader):
        #print(i)
        #print(data)
        # data = data.to(device)
        # label = label.to(device)
        
        # Forward pass
        # data = np.array(data)
        label = Variable(label.float())
        #print(label)
        data = Variable(torch.tensor(data).float())
        data = data.unsqueeze(0)
        #print(data.shape)
        outputs = model(data)
        outputs = outputs[0]
        loss = criterion(outputs, label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            torch.save(model.state_dict(), 'model%d.ckpt'%epoch)
            if loss<min_loss:
                min_loss = loss
                torch.save(model.state_dict(), 'model.ckpt')


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



    #print('Testing MAE : {} '.format(metrics.mean_absolute_error(label, outputs)))

