# import argparse
# from collections import defaultdict
# from pathlib import Path
# import os
# import cv2
# import numpy as np
# from shapely.geometry import Polygon
# from shapely.geometry.point import Point
# from ultralytics import YOLO
# from ultralytics.utils.files import increment_path
# from ultralytics.utils.plotting import Annotator, colors
# import pickle
#
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#
# track_history = defaultdict(list)
# counting_regions = []
# def regionconversion(region_path):
#
#     try:
#         regions = region_path
#         print(regions)
#         with open(regions, 'rb') as f:
#             parked_car_boxes = pickle.load(f)
#         i=1
#         # counting_regions=[]
#         for parked_car_box in parked_car_boxes:
#             # Assuming parked_car_box is a list of four points (x1, y1, x2, y2)
#             polygon_points = [(point[0], point[1]) for point in parked_car_box]  # Extract x, y coordinates
#
#             # Create a new region dictionary with appropriate data
#             new_region = {
#                 "name": f"Parked Region {i}",  # Adjust name as needed
#                 "polygon": Polygon(polygon_points),
#                 "counts": 0,
#                 "dragging": False,
#                 "region_color": (255, 165, 0),
#                 "text_color": (0, 0, 0),
#             }
#             i+=1
#             # Append the new region to counting_regions
#             counting_regions.append(new_region)
#
#     except FileNotFoundError:
#         print("Error: 'regions.p' file not found. No parked car boxes loaded.")
#         current_region = None
#     return counting_regions
# def run(source,region):
#     device = 0
#     view_img = "true"
#     weights = "yolov8n.pt"
#     line_thickness = 2
#     track_thickness = 2
#     region_thickness = 2
#     exist_ok="true"
#     vid_frame_count = 0
#     counting_regions = region
#     # Check source path
#     if not Path(source).exists():
#         raise FileNotFoundError(f"Source path '{source}' does not exist.")
#
#     # Setup Model
#     model = YOLO(f"{weights}")
#     model.to("cuda") if device == "0" else model.to("cpu")
#
#     # Extract classes names
#     names = model.model.names
#
#     # Video setup
#     videocapture = cv2.VideoCapture(source)
#     frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
#     fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
#     frame_width += 350
#     # Output setup
#     save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width , frame_height))
#     overallcount=0
#     total = 0
#     string = ""
#     # Iterate over video frames
#     while videocapture.isOpened():
#         success, frame = videocapture.read()
#         if not success:
#             break
#         vid_frame_count += 1
#
#         # Extract the results
#         results = model.track(frame , persist=True)
#
#         if results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()
#             clss = results[0].boxes.cls.cpu().tolist()
#
#             annotator = Annotator(frame, line_width=line_thickness, example=str(names))
#
#             for box, track_id, cls in zip(boxes, track_ids, clss):
#                 annotator.box_label(box, str(names[cls]), color=colors(cls, True))
#                 bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
#
#                 track = track_history[track_id]  # Tracking Lines plot
#                 track.append((float(bbox_center[0]), float(bbox_center[1])))
#                 if len(track) > 30:
#                     track.pop(0)
#                 points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)
#
#                 # Check if detection inside region
#                 for region in counting_regions:
#
#                     if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
#                         region["counts"] += 1
#                         overallcount +=1
#             total = len(counting_regions)
#             print(overallcount)
#             x1=10
#             y1=55
#             string = str(overallcount)+"/"+str(total)
#             cv2.putText(frame, str(string), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 2.5, (255, 0, 0), 3)
#         # Draw regions (Polygons/Rectangles)
#         for region in counting_regions:
#             region_label = str(region["counts"])
#             region_color = region["region_color"]
#             region_text_color = region["text_color"]
#             region_name = str(region["name"])
#             polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
#             centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)
#             text_size, _ = cv2.getTextSize(
#                 region_name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
#             )
#             text_x = centroid_x - text_size[0] // 2
#             text_y = centroid_y + text_size[1] // 2
#             cv2.rectangle(
#                 frame,
#                 (text_x - 5, text_y - text_size[1] - 5),
#                 (text_x + text_size[0] + 5, text_y + 5),
#                 region_color,
#                 -1,
#             )
#             cv2.putText(
#                 frame, region_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
#             )
#             cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)
#
#         if view_img:
#             if vid_frame_count == 1:
#                 cv2.namedWindow("Parking Pixel")
#                 # cv2.setMouseCallback("Parking Pixel Region Counter Movable", mouse_callback)
#             cv2.imshow("Parking Pixel", frame)
#
#         for region in counting_regions:  # Reinitialize count for each region
#             region["counts"] = 0
#         overallcount=0
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
#     del vid_frame_count
#     video_writer.release()
#     videocapture.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("region", type=str, help="region file path")
#     parser.add_argument("--source", type=str, help="video file path")
#     args = parser.parse_args()
#
#     region_source = regionconversion(args.region)
#     run(args.source, region_source)
import argparse
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
import pickle

def load_regions(region_path):
    try:
        with open(region_path, 'rb') as f:
            regions = pickle.load(f)
        return regions
    except FileNotFoundError:
        print("Error: 'regions.p' file not found. No parked car boxes loaded.")
        return []

def process_video(source, regions, weights="yolov8n.pt", device="cuda", view_img=True):
    model = YOLO(weights).to(device)

    video_capture = cv2.VideoCapture(source)
    frame_width, frame_height = int(video_capture.get(3)), int(video_capture.get(4))
    fps, fourcc = int(video_capture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
    frame_width += 350

    save_dir = increment_path("ultralytics_rc_output/exp", exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    counting_regions = []
    for i, region in enumerate(regions, start=1):
        polygon_points = [(point[0], point[1]) for point in region]
        counting_regions.append({
            "name": f"Parked Region {i}",
            "polygon": Polygon(polygon_points),
            "counts": 0,
            "region_color": (255, 165, 0),
            "text_color": (0, 0, 0)
        })

    overall_count = 0

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

                for region in counting_regions:
                    if region["polygon"].contains(Point(bbox_center)):
                        region["counts"] += 1
                        overall_count += 1

        for region in counting_regions:
            region["counts"] = 0

        if view_img:
            cv2.imshow("Parking Pixel", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video_writer.release()
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("region", type=str, help="region file path")
    parser.add_argument("--source", type=str, help="video file path")
    args = parser.parse_args()

    regions = load_regions(args.region)
    process_video(args.source, regions)
