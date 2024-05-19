import pickle
regions="regions.p"
with open(regions, 'rb') as f:
    parked_car_boxes = pickle.load(f)
print(parked_car_boxes)
import argparse
from collections import defaultdict
from pathlib import Path
from shapely import Polygon
import numpy as np
from ultralytics import YOLO
from shapely.geometry.point import Point

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on an image
results = model.track('carvideo/carPark.mp4', show = True ,save= True,classes=2)  # results list
print("hihihihihhihihihihihhihi")
if results[0].boxes.id is not None:
    boxes = results[0].boxes.xywh.int().cpu().tolist()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    print(boxes)
    print(len(boxes))
    print(clss)
    new_car_boxes = []
    for box in boxes:
        y1 = box[0]
        x1 = box[1]
        y2 = box[2]
        x2 = box[3]
        print(y1)
        p1 = (x1, y1)
        print(p1)
        p2 = (x2, y1)
        p3 = (x2, y2)
        p4 = (x1, y2)
        new_car_boxes.append([p1, p2, p3, p4])
print(new_car_boxes)
# Define multiple polygon point lists
# polygon_points_list = [
#     [(100, 300), (500, 300), (500, 600), (100, 600)],
#     [(600, 100), (800, 100), (800, 400), (600, 400)],
#     # Add more polygon point lists here
# ]
current_region = None
# Create new regions with list comprehension
new_regions = [
    {
        "name": f"Custom Polygon Region {i+1}",  # Generate names
        "polygon": Polygon(points),
        "counts": 0,
        "dragging": False,
        "region_color": (0, 255, 0),  # Change base color (BGR)
        "text_color": (255, 255, 255),
    }
    for i, points in enumerate(new_car_boxes)
]
counting_regions = [
    {
        "name": "YOLOv8 Polygon Region",
        "polygon": Polygon([(230, 17), (768, 17), (768, 801), (230, 801)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),  # BGR Value
        "text_color": (255, 255, 255),  # Region Text Color
    }
]
# Extend the counting_regions list with new regions
counting_regions.extend(new_regions)
print (counting_regions)
