"""Explicit inverse letterbox, with no legacy module imports or weight loading at import."""
import numpy as np

def inverse_letterbox(points,scale,pad):
    if scale<=0: raise ValueError('invalid scale')
    return (np.asarray(points,float)-np.asarray(pad,float))/scale

class YoloDetector:
    def __init__(self,weights):
        from ultralytics import YOLO
        self.model=YOLO(str(weights))
    def __call__(self,image):
        # Ultralytics Results.xy and boxes.xyxy already restored to original image.
        r=self.model.predict(image,imgsz=640,conf=.25,verbose=False,save=False)[0]
        points=r.keypoints.xy.cpu().numpy(); confidence=r.keypoints.conf.cpu().numpy()
        return r.boxes.xyxy.cpu().numpy(),points,confidence,confidence>=.5
