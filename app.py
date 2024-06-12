from kivy.app import App
from kivy.uix.checkbox import CheckBox
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


class SayHello(App):
    def build(self):
        # returns a window object with all it's widgets
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.6, 0.7)
        self.window.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        # image widget
        self.window.add_widget(Image(source="carimage\\bmw2.jpg"))

        # label widget
        self.greeting = Label(
            text="Welcome",
            font_size=18,
            color='#00FFCE'
        )
        self.window.add_widget(self.greeting)
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

        self.button1 = Button(
            text="Click heredasda",
            size_hint=(1, 0.5),
            bold=True,
            background_color='#00FFCE',
            disabled=True
            
            # remove darker overlay of background colour
            # background_normal = ""
        )

        self.button1.bind(on_press=self.callback1)
        self.window.add_widget(self.button1)

        return self.window

    def callback(self, instance):
        # change label text to "Hello + user name!"
        file_path = select_video_file()
        print(file_path)

    def callback1(self, instance):
        # change label text to "Hello + user name!"
        # file_path = select_video_file()
        # print(file_path)
        print("button 2")



# run Say Hello App Calss
if __name__ == "__main__":
    SayHello().run()
