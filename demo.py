from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov9c.pt')

# Run inference on an image
results = model('carvideo//sample.mp4' ,show=True)  # results list
# print("hihihihihhihihihihihhihi")
# results[0]
# results[]
# results["boxes"]
# # View results
# # for r in results:
# #     print(r.boxes)  # print the Boxes object containing the detection bounding boxes
import torch
from torch import classes
'''
from ultralytics import YOLO
from shapely.geometry import Polygon as shapely_poly
from shapely.geometry import box
import argparse
import pickle
import cv2
import numpy as np
# Load the YOLOv8 model
# model = YOLO('yolov8n.yaml', model='yolov8n.pt')
model = YOLO('yolov8n.pt')
# Set the model to evaluation mode
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# model.eval()
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
def get_cars(boxes, class_ids):
    cars = []
    for i in range(boxes.shape[0]):
        if class_ids[i] in [3, 8, 6]:
            cars.append(boxes[i])
    return np.array(cars)
def compute_overlaps(parked_car_boxes, car_boxes):
    new_car_boxes = []
    for box in car_boxes:
        y1 = box[0]
        x1 = box[1]
        y2 = box[2]
        x2 = box[3]

        p1 = (x1, y1)
        p2 = (x2, y1)
        p3 = (x2, y2)
        p4 = (x1, y2)
        new_car_boxes.append([p1, p2, p3, p4])

    overlaps = np.zeros((len(parked_car_boxes), len(new_car_boxes)))
    for i in range(len(parked_car_boxes)):
        for j in range(len(car_boxes)):
            pol1_xy = parked_car_boxes[i]
            pol2_xy = new_car_boxes[j]
            polygon1_shape = shapely_poly(pol1_xy)
            polygon2_shape = shapely_poly(pol2_xy)

            polygon_intersection = polygon1_shape.intersection(
                polygon2_shape).area
            polygon_union = polygon1_shape.union(polygon2_shape).area
            IOU = polygon_intersection / polygon_union
            overlaps[i][j] = IOU
    return overlaps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Video file")
    parser.add_argument('regions_path', help="Regions file",
                        default="regions.p")
    args = parser.parse_args()

    regions = args.regions_path
    with open(regions, 'rb') as f:
        parked_car_boxes = pickle.load(f)

    VIDEO_SOURCE = args.video_path
    alpha = 0.6
    video_capture = cv2.VideoCapture(VIDEO_SOURCE)
    video_FourCC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter("out_test.avi", video_FourCC, video_fps, video_size)

    while video_capture.isOpened():
        success, frame = video_capture.read()
        overlay = frame.copy()
        if not success:
            break

        # Convert the frame to a tensor
        # frame_tensor = torch.from_numpy(frame.transpose((2, 0, 1))).float()
        # Run the model on the frame
        # results = model(frame_tensor.unsqueeze(0))
        # Get the car detections
        # print(results[0]['boxes'].shape)
        # cars = get_cars(results[0]['boxes'], class_list)

        results = model.track(frame, classes=2)
        # if results[0].boxes.id is not None:

        cars = results[0].boxes.xyxy.int().cpu().tolist()
        overlaps = compute_overlaps(parked_car_boxes, cars)

        for parking_area, overlap_areas in zip(parked_car_boxes, results):
            max_IoU_overlap = np.max(overlap_areas)
            if max_IoU_overlap < 0.15:
                cv2.fillPoly(overlay, [np.array(parking_area)], (71, 27, 92))
                free_space = True
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.imshow('output', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    print("output saved as out.avi")
'''