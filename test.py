import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
i = int(input("enter the carparking capacity : "))

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('S:\college\python\detect\carvideo\carv.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)
count = 0

while True:
    list = []
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if frame is None:
        print("Error: Unable to load image")
        continue
    else:
        # Resize the image
        resized_frame = cv2.resize(frame, (1020, 500))

    # frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    print(a)
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
            list.append([c])
    k = len (list)
    if i>k:
        print (i-k ,'vacant space')

    elif i <= k :
        print ('parking is full')

    cv2.putText(frame, str(k), (100, 170), cv2.FONT_HERSHEY_COMPLEX, 5, (255, 0, 0), 3)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
