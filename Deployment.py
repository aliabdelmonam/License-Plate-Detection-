import numpy as np 
import streamlit as st
from ultralytics import YOLO
from pp import img_deploy,vid_play
from PIL import Image
import pandas as pd 
import easyocr
import os
import subprocess
import time

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
        array_text = np.array(array_text)
        df_text = pd.DataFrame(array_text,columns=['License Name'])
        st.image(img,caption='License Image')
        st.write(df_text)
    elif 'video' in file_type:
        temp_video_path = os.path.join("model_video", uploaded_file.name)
        try:
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success("Video uploaded successfully!")
            st.write("Extracting text from video... This may take a while.")
            output_model_path =vid_play(temp_video_path,'test1.mp4')
            st.success("Video processing complete!")
            input_video = output_model_path
            output_video = 'test2.mp4'

            # Construct the FFmpeg command
            command = [
                'ffmpeg',
                '-i', input_video,
                '-vcodec', 'libx264',
                output_video
            ]

            # Execute the command
            try:
                st.write('Starting Convertion')
                subprocess.run(command, check=True)
                st.success(f"Successfully converted '{input_video}' to '{output_video}'.")
                st.write(output_video)
                st.video(output_video)  
                st.download_button("Download Processed Video", data=open(output_model_path, "rb").read(), file_name="processed_video.mp4")
            except Exception as e:
                st.write('An error occurred while processing the video with FFmpeg."An error occurred while processing the video with FFmpeg.')                
        except Exception as e:
            st.error(f"An error occurred while writing the video: {e}")
        
