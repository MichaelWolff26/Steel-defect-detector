import torchvision
import os
import torch
import pandas as pd

def create_label_list(data_path):
    list_names=[]
    list_target=[]
    for f in os.listdir(data_path+"\\good"):
        list_names.append("\\good\\"+f)
        list_target.append(0)
    for f in os.listdir(data_path+"\\defect"):
        list_names.append("\\defect\\"+f)
        list_target.append(1)

    df1=pd.DataFrame(list_names,columns=["file_name"])
    df1["label"]=pd.DataFrame(list_target)
    return df1




def DataReader(data_path): # Reading dataframe and image-data, return x_batch, y_batch as torch tensors
    df1=create_label_list(data_path=data_path)
    img_size=torchvision.io.read_image(data_path+df1["file_name"][1]).size()  # get image size
    x_batch_size=2*len(df1) # x_batch_size is 2 times the length of the original data, because of the flipping operation
    x_batch=torch.zeros(x_batch_size,img_size[0],img_size[1],img_size[2], dtype=torch.float32) # creating x_batch
    y_batch=torch.zeros(x_batch_size, dtype=torch.long) # creating y_batch

    c_var=0

    for i in range(len(df1)):   # read the images to x_batch, crop them to size, normalize the values (/255) and flip them; assign labels to y_batch
        
        img_temp=torchvision.io.read_image(data_path+df1["file_name"][i])
        img_cropped=torchvision.transforms.functional.crop(img_temp,top=0, left=0, height=633, width=228)
        norm_img_temp=img_cropped/255
        x_batch[c_var]=norm_img_temp
        y_batch[c_var]=df1["label"][i]
        x_batch[c_var+1]=torchvision.transforms.functional.hflip(norm_img_temp)
        y_batch[c_var+1]=df1["label"][i]
        c_var=c_var+2
    return x_batch,y_batch,df1