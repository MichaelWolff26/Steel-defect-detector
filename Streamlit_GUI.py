import streamlit 
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.models as models
import os
import torch

streamlit.header("Steel surface defect analyser")
number=streamlit.slider("Pick one of the 100 sample images")


curr_path=os.getcwd()
data_subdir_path="\\Dataset-trafo\\"
data_path=curr_path+data_subdir_path

list_namesg=[]
list_namesd=[]
list_samples=[]
for f in os.listdir(data_path+"\\good"):
    list_namesg.append("\\good\\"+f)
for f in os.listdir(data_path+"\\defect"):
    list_namesd.append("\\defect\\"+f)
for i in range (50):
    list_samples.append(list_namesg[i])
    list_samples.append(list_namesd[i])
data2=torchvision.io.read_image(data_path+list_samples[number])
streamlit.write(data2.shape)

data=streamlit.file_uploader("Or upload own surface image",type="png")
own_data=streamlit.toggle("Use own surface image")




if (data or data2) is not None:
    if data is not None:
        image=Image.open(data)
        convert_tensor = torchvision.transforms.ToTensor()
        test=convert_tensor(image)
    if data2 is not None and not own_data:
        test=data2
    
    test1=test.permute(2,1,0)
    test_numpy=test1.numpy()
      
    
    streamlit.image(test_numpy)
    
    if streamlit.button("Analyze"):
        img_cropped=torchvision.transforms.functional.crop(test,top=0, left=0, height=633, width=228)
        norm_img=img_cropped/255
        streamlit.write(norm_img.shape)
        norm_img_2=norm_img[None,:,:,:]
        model=models.resnet.ResNet(models.resnet.BasicBlock, [2, 2, 2, 2],num_classes=2)
        
        path_mp=curr_path+"\\model_parameters\\model.md"
        model=torch.load(path_mp)
        pred=model(norm_img_2)
        class_var=torch.argmax(pred, dim=1)

        c1,c2=streamlit.columns(2)
        c1.header("Surface Quality:")
        if class_var==0:
            c2.image("Streamlit_data\\IO.png",width=140)
        if class_var==1:
            c2.image("Streamlit_data\\NIO.png",width=140)


    
   


