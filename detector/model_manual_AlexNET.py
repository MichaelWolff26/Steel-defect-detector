# Manual AlexNET based approach
import torch.nn as nn
class AlexNET(nn.Module):
    def __init__(self):
        super(AlexNET,self).__init__()
        self.convlayer1=nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.relu1=nn.ReLU()
        self.max_pool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        self.convlayer2=nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=0)
        self.relu2=nn.ReLU()
        self.max_pool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        self.convlayer3=nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding="same")
        self.relu3=nn.ReLU()
        self.convlayer4=nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding="same")
        self.relu4=nn.ReLU()
        self.convlayer5=nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding="same")
        self.relu5=nn.ReLU()
        self.max_pool3=nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        self.flatten=nn.Flatten()
        
        self.dense1=nn.Linear(in_features=4096,out_features=1536)
        self.relu6=nn.ReLU()
        self.dropout1=nn.Dropout(0.5)
        self.dense2=nn.Linear(in_features=1536,out_features=1536)
        self.relu7=nn.ReLU()
        self.dropout2=nn.Dropout(0.5)
        self.dense3=nn.Linear(in_features=1536,out_features=2)
        self.relu8=nn.ReLU()
        
    def forward(self,X):
        X=self.convlayer1(X)
        X=self.relu1(X)
        X=self.max_pool1(X)
        X=self.convlayer2(X)
        X=self.relu2(X)
        X=self.max_pool2(X)
        X=self.convlayer3(X)
        X=self.relu3(X)
        X=self.convlayer4(X)
        X=self.relu4(X)
        X=self.convlayer5(X)
        X=self.relu5(X)
        X=self.max_pool3(X)
        X=self.flatten(X)
        X=self.dense1(X)
        X=self.relu6(X)
        X=self.dropout1(X)
        X=self.dense2(X)
        X=self.relu7(X)     
        X=self.dropout2(X)
        X=self.dense3(X)
        X=self.relu8(X)
        
        return X
    
    def AlexNET():
        model=AlexNET()
        return model
        