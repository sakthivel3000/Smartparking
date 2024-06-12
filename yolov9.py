from ultralytics import YOLO
# from set_regions import sel
# Load a model
model = YOLO("best.pt")  # load a pretrained model (recommended for training)
# # This might not work as expected
# # model = YOLO("use_WongKinYiu/yolov9_train.pt")
results = model.track(source="S:\college\python\detect\carvideo\sample.mp4" ,show=True,persist=False)
