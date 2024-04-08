# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
# results = model(source="carimage/Figure_1.png", show = True  ,save=True)
# # from ultralytics import YOLO
# #
# # # Build a YOLOv9c model from scratch
# # model = YOLO('yolov9c.yaml')
# #
# # # Build a YOLOv9c model from pretrained weight
# # model = YOLO('yolov9c.pt')
# #
# # # Display model information (optional)
# # model.info()
# #
# # # Train the model on the COCO8 example dataset for 100 epochs
# # results = model.train(data='coco8.yaml', epochs=100, imgsz=640)
# #
# # # Run inference with the YOLOv9c model on the 'bus.jpg' image
# # results = model('path/to/bus.jpg')
import argparse
import cv2
import numpy as np
import os
import pickle
from ultralytics import YOLO

def get_cars(boxes, class_ids):
    cars = []
    for i, box in enumerate(boxes):
        if class_ids[i] in [2, 5, 7]:  # Adjust class IDs for cars in YOLOv8
            cars.append(box)
    return np.array(cars)

def compute_overlaps(parked_car_boxes, car_boxes):
    # Your implementation for computing overlaps between boxes
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Video file")
    parser.add_argument('regions_path', help="Regions file", default="regions.p")
    args = parser.parse_args()

    regions = args.regions_path
    with open(regions, 'rb') as f:
        parked_car_boxes = pickle.load(f)

    VIDEO_SOURCE = args.video_path
    alpha = 0.6
    video_capture = cv2.VideoCapture(VIDEO_SOURCE)
    video_FourCC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter("out_test.avi", video_FourCC, video_fps, video_size)

    # Load YOLOv8 model
    # yolov8_model = YoloV8()  # Assuming yolov8 is a package providing YOLOv8 implementation
    # yolov8_model.load_weights("path_to_yolov8_weights")
    model = YOLO('yolov8n.pt')
    while video_capture.isOpened():
        success, frame = video_capture.read()
        overlay = frame.copy()
        if not success:
            break

        # Preprocess frame for YOLOv8 input
        # Your preprocessing code here

        # Perform inference with YOLOv8
        # results = yolov8_model.detect(frame)
        results = model(frame)

        cars = get_cars(results.boxes, results['classes'])
        overlaps = compute_overlaps(parked_car_boxes, cars)

        for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):
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

