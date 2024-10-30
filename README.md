# License-Plate-Detection

## using `ffmpeg` library to re-encode the video cause `OpenCV` and `skvideo` libraries, which you likely used to generate or manipulate the video, may encode MP4 files using codecs that are not fully supported by the HTML5 video player
## HTML5 video players, which Streamlit uses to display videos, only support certain codecs reliably `H.264` for video
