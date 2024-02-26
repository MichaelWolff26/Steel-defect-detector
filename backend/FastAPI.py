from fastapi import FastAPI,UploadFile, File
import os 
import torchvision
import torch
app = FastAPI()

@app.get("/")
def hello():
    return {"API":"API is working fine"}

@app.post("/upload_image")
async def upload_image(filename:str,img_file:UploadFile =File(...)):
    class_var=0
    class_str="unsucessful" 
    class_str=filename

    if '.jpg' in filename or '.jpeg' in filename or '.png' in filename:
        class_str="passed0" 
        file_save_path="./images_fastapi/"+img_file.filename
        if os.path.exists("./images_fastapi") == False:
            os.makedirs("./images_fastapi")

        with open(file_save_path, "wb") as f:
            f.write(img_file.file.read())

       
        img_tensor=torchvision.io.read_image("./images_fastapi/"+img_file.filename)
        img_cropped=torchvision.transforms.functional.crop(img_tensor,top=0, left=0, height=633, width=228)
        norm_img=img_cropped/255
        norm_img_2=norm_img[None,:,:,:]


        model=torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2],num_classes=2)
        path_mp=os.getcwd()+"/model_parameters/model.md"
        model=torch.load(path_mp)
        pred=model(norm_img_2)
        class_var=torch.argmax(pred, dim=1)
        class_str=int(class_var)

        os.remove("./images_fastapi/"+img_file.filename)
    
    return class_str
