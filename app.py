# import tkinter.filedialog
# import tkinter as tk
# from moviepy.video.io.bindings import mplfig_to_npimage
# import matplotlib.pyplot as plt
# import numpy as np
# from moviepy.editor import *
#
# def select_video_file():
#     root = tk.Tk()
#     root.withdraw()
#     file_path = tk.filedialog.askopenfilename(filetypes=[('Video files', '*.mp4 *.avi *.mkv')])
#     root.destroy()
#     return file_path
#
#
#
# if __name__ == '__main__':
#
#     file_path = select_video_file()
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
import tkinter.filedialog
import tkinter as tk
from ultralytics import YOLO
from moviepy.video.io.bindings import mplfig_to_npimage
import cv2
import pandas as pd
from ultralytics import YOLO


def select_video_file():
    root = tk.Tk()
    root.withdraw()
    file_path = tk.filedialog.askopenfilename(filetypes=[('Video files', '*.mp4 *.avi *.mkv')])
    root.destroy()
    return file_path


def dectect(car, file_path):
    model = YOLO('yolov8n.pt')

    def RGB(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            colorsBGR = [x, y]
            print(colorsBGR)

    # i = int(input("enter the parking capacity : "))
    i = int(car)
    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    cap = cv2.VideoCapture(file_path)

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

        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame)
        #   print(results)
        a = results[0].boxes.data
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                list.append([c])
        k = len(list)


        cv2.putText(frame, str(k), (100, 170), cv2.FONT_HERSHEY_COMPLEX, 5, (255, 0, 0), 3)
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        return k
    cap.release()
    cv2.destroyAllWindows()


class SayHello(App):
    def build(self):
        # returns a window object with all it's widgets
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.6, 0.7)
        self.window.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        # image widget
        self.window.add_widget(Image(source="bmw2.jpg"))

        # label widget
        self.greeting = Label(
            text="Enter the parking capacity",
            font_size=18,
            color='#00FFCE'
        )
        self.window.add_widget(self.greeting)

        # text input widget
        self.user = TextInput(
            multiline=False,
            padding_y=(20, 20),
            size_hint=(1, 0.5)
        )

        self.window.add_widget(self.user)

        # button widget
        self.button = Button(
            text="Click here",
            size_hint=(1, 0.5),
            bold=True,
            background_color='#00FFCE',
            # remove darker overlay of background colour
            # background_normal = ""
        )

        self.button.bind(on_press=self.callback)
        self.window.add_widget(self.button)

        return self.window

    def callback(self, instance):
        # change label text to "Hello + user name!"
        file_path = select_video_file()
        k = dectect(self.user.text, file_path)
        i = int(self.user.text)
        if i > k:
            a = i - k
            self.greeting.text = str(a) + " vacant space"
        elif i <= k:
            self.greeting.text = "parking space is full"


# run Say Hello App Calss
if __name__ == "__main__":
    SayHello().run()
