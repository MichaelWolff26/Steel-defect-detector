# save wrong detections to folders for further analysis
import os
import shutil
from torchvision.utils import save_image

def write_wrong_detections(data_path,x_test,y_test,is_correct):  

    if data_path+"wrong_defect":
        shutil.rmtree(data_path+"wrong_defect\\")
    if data_path+"wrong_io":
        shutil.rmtree(data_path+"wrong_io\\")
    os.mkdir(path=data_path+"wrong_defect\\")
    os.mkdir(path=data_path+"wrong_io\\")
    for i in range(len(is_correct)):
        if is_correct[i]==0 and y_test[i]==0:   # if classified as ok/io but it is a defect, save to wrong_defect folder
            save_image(x_test[i],fp=data_path+"wrong_defect\\"+str(i)+".png")   
        if is_correct[i]==0 and y_test[i]==1:   # if classified as defect but it is ok/io, save to wrong_io folder
            save_image(x_test[i],fp=data_path+"wrong_io\\"+str(i)+".png")    