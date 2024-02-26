import torchvision.models as models

def resnet():
    model=models.resnet.ResNet(models.resnet.BasicBlock, [2, 2, 2, 2],num_classes=2)
    return model