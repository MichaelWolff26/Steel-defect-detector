# Manual inception based approach
import torch.nn as nn
import torch




class inception_block(nn.Module):
    def __init__(self, input_channels):
        super(inception_block,self).__init__()

        self.convlayer2=nn.Conv2d(in_channels=input_channels, out_channels=input_channels//2, kernel_size=1, stride=1, padding="same")
        self.convlayer3=nn.Conv2d(in_channels=input_channels, out_channels=input_channels//2, kernel_size=1, stride=1, padding="same")
        self.convlayer4=nn.Conv2d(in_channels=input_channels, out_channels=input_channels//2, kernel_size=1, stride=1, padding="same")
        self.convlayer5=nn.Conv2d(in_channels=input_channels//2, out_channels=input_channels, kernel_size=3, stride=1, padding="same")
        self.convlayer6=nn.Conv2d(in_channels=input_channels//2, out_channels=input_channels//2, kernel_size=5, stride=1, padding="same")
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.convlayer7=nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1, padding="same")
        self.averagepool=nn.MaxPool2d(kernel_size=7, stride =3, padding=0)

    def forward(self,X):

        X1=self.convlayer4(X)

        X2=self.convlayer2(X)
        X2=self.convlayer5(X2)

        X3=self.convlayer3(X)
        X3=self.convlayer6(X3)

        X4=self.maxpool(X)
        X4=self.convlayer7(X4)

        X=torch.concat((X1,X2,X3,X4), dim=1)
        X=self.averagepool(X)
        

        return X

class inception_Network(nn.Module):
    def __init__(self, inception_block):
        super(inception_Network,self).__init__()
        self.input_conv=nn.Conv2d(in_channels=3, out_channels=32, kernel_size=11, stride=4, padding=0)
        self.inception_block_1=inception_block(input_channels=32)
        self.inception_block_2=inception_block(input_channels=96)
        self.dense1=nn.Linear(in_features=17280, out_features=1000)
        self.relu1=nn.ReLU()
        self.dense2=nn.Linear(in_features=1000, out_features=2)
        self.relu2=nn.ReLU()
        self.flatten=nn.Flatten()
        self.relu3=nn.ReLU()
    def forward(self, X):
        X=self.input_conv(X)
        X=self.inception_block_1(X)
        X=self.inception_block_2(X)
        
        X=self.flatten(X)
        X=self.relu1(X)
        X=self.dense1(X)
        X=self.relu2(X)
        X=self.dense2(X)
        X=self.relu3(X)
        return X
    
def inception():
    model=inception_Network(inception_block=inception_block)
    return model

    
