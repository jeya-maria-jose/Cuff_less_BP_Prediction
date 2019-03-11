import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from custom_data import bpdata_train,bpdata_test,bpdata_val
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics  

bp_train_dataset = bpdata_train(csv_file='/home/jeyamariajose/Projects/dl/bp_train_new.csv',
                                    root_dir='/home/jeyamariajose/Projects/dl/data/cleaned/train')

bp_test_dataset = bpdata_test(csv_file='/home/jeyamariajose/Projects/dl/bp_test_new.csv',
                                    root_dir='/home/jeyamariajose/Projects/dl/data/cleaned/test/')

bp_val_dataset = bpdata_val(csv_file='/home/jeyamariajose/Projects/dl/bp_val_new.csv',
                                    root_dir='/home/jeyamariajose/Projects/dl/data/cleaned/val/')



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 20
num_classes = 7
batch_size = 1
learning_rate = 0.001


train_loader = torch.utils.data.DataLoader(dataset=bp_train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=bp_test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

val_loader = torch.utils.data.DataLoader(dataset=bp_val_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)



class ConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4,kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.fc1 = nn.Linear(32064, 32)
        self.fc2 = nn.Linear(32, 7)
        self.dropout = nn.Dropout(0.3) 
        #self.sm = nn.Softmax()

        
    def forward(self, x):
        out = self.layer1(x)

        out = self.dropout(out)
        #print(out.shape)
        out = self.layer2(out)

        out = self.dropout(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        #out = self.sm(out)
        return out

model = ConvNet(num_classes).to(device)
min_loss = 1000
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)


for epoch in range(num_epochs):
    for i,(data,reg,label) in enumerate(train_loader):
        #print(i)
        #print(data)
        # data = data.to(device)
        # label = label.to(device)
        
        # Forward pass
        # data = np.array(data)
        #label = label.float()

        #ll = label.numpy()

        #hot_out = np.zeros((7,1))
        #hot_out[int(ll[0])-1] =1;
        total = 0
        correct = 0

        #print(label)
        data = torch.tensor(data).float()
        data = data.unsqueeze(0)
        #print(data.shape)
        outputs = model(data)
        #outputs = outputs[0]


        #print(outputs.shape)
        #print(outputs)
        loss = criterion(outputs, label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print("predicted",predicted)
        #print("label",labels)
        #print(predicted)
        
        #print(correct,total)

        
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            torch.save(model.state_dict(), 'model%d.ckpt'%epoch)

            if loss<min_loss:
                min_loss = loss
                torch.save(model.state_dict(), 'model.ckpt')
    total = 0
    correct = 0
    for i,(data,reg,labels) in enumerate(train_loader): 

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

    print('Train Accuracy : {} %'.format(100 * correct / total))

    total = 0
    correct = 0
        
    for i,(data,reg,labels) in enumerate(val_loader):          
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

    print('Validation Accuracy : {} %'.format(100 * correct / total))



