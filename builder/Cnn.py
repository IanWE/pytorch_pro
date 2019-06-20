import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=5,
                    stride=1,
                    padding=2
                    ),
                nn.Relu(),
                nn.MaxPool2d(kernel_size=2)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(16,32,5,1,2),
                nn.Relu(),
                nn.MaxPool2d(2)
                )
        self.out = nn.Linear(32*7*7,10) 
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1) #展平多维的卷积图层
        output = self.out(x)
        return output

cnn = CNN()
print(cnn) #net architecture

optimizer = torch.optim.Adam(cnn.parameters(),lr=0.001)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for tep,(b_x,b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.bachward()
        optimizer.step()
        if step %50 ==0:
            test_output,last_layer = cnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.squeeze()
            accuracy = float(sum(pred_y==test_y)/float(test_y.size(0)))


test_output = cnn(test_x[:10])
pred_y = torch.max(test_output,1)[1].data.numpy().squeeze()


