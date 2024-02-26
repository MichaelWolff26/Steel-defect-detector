# Manual small Conv Network with one residual connection for first testing
import torch.nn as nn
class conv_network(nn.Module):
    def __init__(self):
        super(conv_network,self).__init__()
        self.convlayer1=nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, stride=2, padding=0)
        self.convlayer2=nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=0)
        self.max_pool=nn.MaxPool2d(kernel_size=5,stride=2,padding=0)
        self.drop_out=nn.Dropout(0.2)
        
        self.dense1=nn.Linear(in_features=3168,out_features=512)
        self.dense2=nn.Linear(in_features=512,out_features=512)
        self.dense3=nn.Linear(in_features=512,out_features=128)
        self.dense4=nn.Linear(in_features=128,out_features=2)
    def forward(self,X):
        X=self.convlayer1(X)
        X=nn.functional.relu(X)
        X=self.max_pool(X)
        X=nn.functional.relu(X)
        X=self.convlayer2(X)
        X=nn.functional.relu(X)
        X=self.max_pool(X)
        X=nn.functional.relu(X)
        X=torch.flatten(X,1)
        X=self.dense1(X)
        X=nn.functional.relu(X)
        X=self.drop_out(X)
        residual=X
        X=self.dense2(X)
        X=nn.functional.relu(X)
        X=self.drop_out(X)
        X=X+residual
        X=self.dense3(X)
        X=nn.functional.relu(X)
        X=self.drop_out(X)
        X=self.dense4(X)
        X=nn.functional.relu(X)
        return X
    
def small_conv():
    model=conv_network()
    return model