import pickle

from shapely import Polygon

regions = "regions.p"
# counting_regions = [
#     {
#         "name": "YOLOv8 Polygon Region",
#         "polygon": Polygon([(893, 189),(893, 237),(993, 235),(993, 190)]),  # Polygon points
#         "counts": 0,
#         "dragging": False,
#         "region_color": (255, 42, 4),  # BGR Value
#         "text_color": (255, 255, 255),  # Region Text Color
#     }]
try:
    with open(regions, 'rb') as f:
        parked_car_boxes = pickle.load(f)
    counting_regions =[]
    for parked_car_box in parked_car_boxes:
        # Assuming parked_car_box is a list of four points (x1, y1, x2, y2)
        polygon_points = [(point[0], point[1]) for point in parked_car_box]  # Extract x, y coordinates

        # Create a new region dictionary with appropriate data
        new_region = {
            "name": "Parked Car Region",  # Adjust name as needed
            "polygon": Polygon(polygon_points),
            "counts": 0,
            "dragging": False,
            "region_color": (255, 165, 0),  # Adjust color (BGR)
            "text_color": (0, 0, 0),  # Adjust text color
        }

        # Append the new region to counting_regions
        counting_regions.append(new_region)

except FileNotFoundError:
    print("Error: 'regions.p' file not found. No parked car boxes loaded.")
print(counting_regions)
print("Successfully appended parked car boxes (if any) to counting_regions.")