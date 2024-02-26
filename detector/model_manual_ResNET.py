# Manual ResNET based approach
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,num_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block1=nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding="same")
        
        self.batch_norm1=nn.BatchNorm2d(num_features=num_channels)
        self.conv_block1_1=nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding="same")
        self.batch_norm2=nn.BatchNorm2d(num_features=num_channels)
        self.RELU1=nn.ReLU()
        self.RELU2=nn.ReLU()
    def forward(self,X):
        Residual=X
        X=self.conv_block1(X)
        X=self.batch_norm1(X)
        X=self.RELU1(X)
        X=self.conv_block1_1(X)
        X=self.batch_norm2(X)
        X=X+Residual
        X=self.RELU2(X)
        return X



class ResNET(nn.Module):
    def __init__(self,Residualblock,num_block1,num_block2,num_block3):
        super(ResNET,self).__init__()
        self.conv_transition0=nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=0)
        self.max_pool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        self.block1_list=[]
        for i in range(num_block1):
            self.block1_list.append(Residualblock(num_channels=32))
        self.block1=nn.Sequential(*self.block1_list)
        self.conv_transition1=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=0)
        self.block2_list=[]
        for i in range(num_block2):
            self.block2_list.append(Residualblock(num_channels=64))
        self.block2=nn.Sequential(*self.block2_list)
        self.conv_transition2=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=2, padding=0)
        self.block3_list=[]
        for i in range(num_block3):
            self.block3_list.append(Residualblock(num_channels=128))
        self.block3=nn.Sequential(*self.block3_list)
        self.avg_pool=nn.AvgPool2d(kernel_size=3,stride=2,padding=0)
        self.dense1=nn.Linear(in_features=8704, out_features=2000)
        self.relu1=nn.ReLU()
        self.dense2=nn.Linear(in_features=2000, out_features=2)
        self.relu2=nn.ReLU()
        

    def forward(self,X):
        X=self.conv_transition0(X)
        X=self.max_pool1(X)
        X=self.block1(X)
        X=self.conv_transition1(X)
        X=self.block2(X)
        X=self.conv_transition2(X)
        X=self.block3(X)
        X=self.avg_pool(X)
        X=nn.Flatten()(X)
        X=self.dense1(X)
        X=self.relu1(X)
        X=self.dense2(X)
        X=self.relu2(X)
        return X
    
def manual_resnet():
    model=ResNET()
    return model
    