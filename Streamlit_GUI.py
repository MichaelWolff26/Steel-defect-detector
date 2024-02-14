import streamlit 
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

streamlit.header("Steel surface defect analyser")
number=streamlit.slider("Pick one of the 100 sample images")
data=streamlit.file_uploader("Or upload own surface image",type="png")


if data is not None:
    image=Image.open(data)
    convert_tensor = torchvision.transforms.ToTensor()
    test=convert_tensor(image)
    
    test1=test.permute(2,1,0)
    test_numpy=test1.numpy()
    
    #fig=plt.imshow(test.permute(1,2,0))
    #plt.savefig('x',dpi=600)
    
    
    streamlit.image(test_numpy)
    
    if streamlit.button("Analyze"):
        c1,c2=streamlit.columns(2)
        c1.header("Surface Quality:")
        c2.image("Streamlit_data\\NIO.png",width=140)


    
   


