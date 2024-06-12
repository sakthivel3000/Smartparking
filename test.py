"""import cv2
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
cv2.destroyAllWindows() """

"""def remove_extension(filename):
  
  Removes the file extension from a filename.

  Args:
      filename (str): The filename with the extension.

  Returns:
      str: The filename without the extension.
  """
#   dot_index = filename.rfind(".")  # Find the last index of '.' (for extension)
#   if dot_index > 0:  # Check if there's an extension
#     return filename[:dot_index]  # Return everything before the last '.'
#   else:
#     return filename  # Return the original filename if no extension
#
# # Example usage
# filename = "/home/user/data/myfile.txt/myfile.txt"
# filename_without_ext = remove_extension(filename)
# print(filename_without_ext)  # Output: myfile

import os
import pickle
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup

# Function to check for corresponding pickle file
def has_corresponding_pickle(video_filepath):
    pickle_filepath = os.path.splitext(video_filepath)[0] + ".p"
    return os.path.exists(pickle_filepath)

# Function to set regions (implementation details needed)
def set_regions():
    # Replace this with your actual region setting logic
    print("Setting regions...")
    # ... your region setting code ...

# Function to detect objects (implementation details needed)
def detect():
    # Replace this with your actual object detection logic
    print("Detecting objects...")
    # ... your object detection code ...

class VideoApp(App):
    def __init__(self, video_filepath):
        super().__init__()  # Call parent class constructor
        self.video_filepath = video_filepath  # Assign video_filepath to self


    def build(self):
        # No argument needed, logic within build

        # Check for corresponding pickle file
        has_pickle = has_corresponding_pickle(self.video_filepath)

        layout = BoxLayout(orientation="vertical")

        if has_pickle:
            # Show button if pickle file exists
            print("in the if b")

            # button_label = "Load Regions from Pickle (.p)"
            # button = Button(text=button_label, on_press=self.load_regions)
            # layout.add_widget(button)
            print("in the if a")
        else:
            # Call set_regions and then detect if no pickle file
            print("in the else b")
            set_regions()
            detect()


        return layout

    def load_regions(self, instance):
        # Load regions from pickle file
        print("hiii")
        pickle_filepath = os.path.splitext(self.video_filepath)[0] + ".p"
        with open(pickle_filepath, 'rb') as f:
            regions = pickle.load(f)

        # Call detect after loading regions (adjust based on your logic)
        detect()


    def ask_add_region(self):
        # Popup to ask for adding a new region
        content = Label(text="Do you want to add a new region?")
        yes_button = Button(text="Yes", on_press=self.set_regions_and_detect)
        no_button = Button(text="No", on_press=self.detect_only)
        popup = Popup(title="Add Region?", content=content,
                       buttons=[yes_button, no_button])
        popup.open()

    def set_regions_and_detect(self, instance):
        # Call set_regions and then detect after user selects "Yes"
        set_regions()
        detect()

    def detect_only(self, instance):
        # Call detect only if user selects "No"
        detect()

# Run the app with a video filepath (replace with your actual video)
if __name__ == "__main__":
    video_filepath = "S:\college\python\detect\carvideo\cartesting.mp4"  # Replace with your video path
    VideoApp(video_filepath).run()  # Only the video filepath argument is passed

