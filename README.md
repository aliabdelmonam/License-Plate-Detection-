# License Plate Detection
## Overview 
This project enables accurate detection and recognition of license plates from video and image data. By leveraging **`YOLO`** for object detection and **`PaddleOCR`** for optical character recognition (OCR), the system can locate and interpret license plates efficiently. To enhance video processing reliability, **`FFmpeg`** is used to handle video inputs, addressing certain limitations encountered when using OpenCV alone.
## Workflow
The License Plate Recognition Project is a robust and efficient system designed for detecting and reading license plates in images or from recorded video. This project combines:
-  **YOLOV8n** (You Only Look Once) for fast and accurate license plate detection.
-  **PaddleOCR** for reliable text recognition, supporting multiple languages.
-  **FFmpeg** re-encode the video cause `OpenCV` and `skvideo` libraries, which you likely used to generate or manipulate the video, may encode MP4 files using codecs that are not fully supported by the HTML5 video player TML5 video players, which Streamlit uses to display videos, only support certain codecs reliably `H.264` for video

 ## Componenet
 - `best.pt`: model weights **YOLOv8n**
 - `Deployment.py`: model deployment
 - `pp.pt`: functions
 
