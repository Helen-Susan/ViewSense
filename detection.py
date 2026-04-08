import torch
import cv2
from yolov5.utils.general import non_max_suppression

model = torch.hub.load('ultralytics/yolov5', 'custom', path='ViewSense\\results_yolov5\\best.pt', force_reload=True)
img = cv2.imread('ViewSense\\Screenshot 2026-04-08 003219.png')
img = torch.from_numpy(img).unsqueeze(0)
img = img.permute(0, 3, 1, 2) / 255.0
pred = model(img)
pred = non_max_suppression(pred)   