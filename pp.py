import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from ultralytics import YOLO
import easyocr
from PIL import Image
from paddleocr import PaddleOCR

def get_points(result):
  li = []
  for box in result[0]:
    x1,y1,x2,y2 =  map(int,box.boxes.xyxy[0].tolist())
    temp = list()
    temp.append(x1)
    temp.append(y1)
    temp.append(x2)
    temp.append(y2)
    li.append(temp)
  return li


# def license_text(reader,img):
#   tt = reader.readtext(img)
#   txt_list = list()
#   for temp in tt:
#     _,text,_ = temp
#     txt_list.append(text)
#   return  " ".join(txt_list).upper()
def license_text(reader,img):
  text = reader.ocr(img,cls=True)
  return text


def vid_play(vid_link,name_of_vid):
    reader = easyocr.Reader(['en'])
    cap = cv2.VideoCapture(vid_link)
    model = YOLO('best.pt')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize VideoWriter to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    path = os.path.join('model_video',name_of_vid)
    out = cv2.VideoWriter(path, fourcc, fps, (frame_width, frame_height))
    while  cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

        result=model.predict(source=frame,conf=.5)

        obj_list = get_points(result)
        
        for obj in obj_list:
            x1,y1,x2,y2=obj
            ff = gray[y1:y2,x1:x2]
            ff = cv2.resize(ff,(400,150))
            adaptive_image = cv2.adaptiveThreshold(ff,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,71,1)
            text = license_text(reader,adaptive_image)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(225,0,0),2)
            org = (x1, y1 - 15)  # Position text below the box
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 4, cv2.LINE_AA)
        out.write(frame)
    cap.release()
    out.release()
    return path

def img_deploy(img):
    ocr = PaddleOCR(use_angle_cls=True,lang='en')
    model = YOLO('best.pt')
    result = model.predict(source=img, conf=0.5)
    obj_list = get_points(result)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    text_list=[]
    for obj in obj_list:
        x1, y1, x2, y2 = obj  # Bounding box coordinates

        license = img[y1:y2, x1:x2]
        text = license_text(ocr,license)      
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 225, 0), 3)
        if text[0] is not None :
            text = str(text[0][0][1][0])
            text_list.append(text)
            org = (x1, y1 - 15)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return text_list,img

