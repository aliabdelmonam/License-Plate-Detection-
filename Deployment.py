import numpy as np 
import streamlit as st
from ultralytics import YOLO
from pp import img_deploy
from PIL import Image
import pandas as pd 
import easyocr

st.title("License Plate Extraction")
model = YOLO('best.pt')
reader = easyocr.Reader(['en'])


uploaded_file = st.file_uploader("Enter License")


if uploaded_file is not None:
    # You can now process the uploaded file
    file_type = uploaded_file.type

    if 'image' in file_type:
        image = np.array(Image.open(uploaded_file))  
        array_text,img=img_deploy(image)
        df_text = pd.Series(array_text,name=['License Text'])
        st.image(img)
        st.dataframe(df_text)
    # elif 'video' in file_type:
        
