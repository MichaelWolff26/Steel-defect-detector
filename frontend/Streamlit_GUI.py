import streamlit 
from PIL import Image
import os
import requests
from pathlib import Path


streamlit.header("Steel surface defect analyser")
number=streamlit.slider("Pick one of the 100 sample images")


curr_path=os.getcwd()
data_subdir_path="/Dataset-trafo"
data_path=curr_path+data_subdir_path

list_namesg=[]
list_namesd=[]
list_samples=[]
for f in os.listdir(data_path+"/good"):
    list_namesg.append("/good/"+f)
for f in os.listdir(data_path+"/defect"):
    list_namesd.append("/defect/"+f)
for i in range (50):
    list_samples.append(list_namesg[i])
    list_samples.append(list_namesd[i])
data2=Image.open(data_path+list_samples[number])

data=streamlit.file_uploader("Or upload own surface image",type="png")



own_data=streamlit.toggle("Use own surface image")

if (data or data2) is not None:
    temp_save_path=None
    if data is not None and own_data:
        test=Image.open(data)
        file=data.getvalue()                   
        filename=data.name
    if data2 is not None and not own_data:
        test=data2
        filename=data_path+list_samples[number]
        file=open(data_path+list_samples[number],"rb")

    if not (own_data and data==None):
        streamlit.image(test)
    
    if streamlit.button("Analyze"):
        url = "http://backend:8000/upload_image"
        payload = {"filename": filename}
        response=requests.post(url,params=payload,files={"img_file":file})
        class_var=response.content.decode()
        if temp_save_path is not None:
            file.close()
            os.remove(temp_save_path)
            temp_save_path=None

        c1,c2=streamlit.columns(2)
        c1.header("Surface Quality:")
                
        if class_var=="0":
            c2.image("Streamlit_data/IO.png",width=140)
        if class_var=="1":
            c2.image("Streamlit_data/NIO.png",width=140)
        
streamlit.write("The Data is based on the Kolektor Surface-Defect Dataset 2 and is modified")
streamlit.write("License: https://creativecommons.org/licenses/by-nc-sa/4.0/")
streamlit.write("This is a non commercial website")

    
   


