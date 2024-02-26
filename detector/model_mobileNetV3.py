import torchvision.models as models
def mobilenet():
    model=models.mobilenet.mobilenet_v3_small(num_classes=2)
    return model
