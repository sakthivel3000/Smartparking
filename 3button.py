# from kivy.app import App
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.button import Button
# from kivy.properties import BooleanProperty
#
#
# class MyWidget(BoxLayout):
#     condition_met = BooleanProperty(False)
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.orientation = 'vertical'
#
#         # Button 1
#         self.btn1 = Button(text='Button 1', on_press=self.check_condition)
#         self.add_widget(self.btn1)
#
#         # Button 2 (initially invisible)
#         self.btn2 = Button(text='Button 2')
#         self.btn2.opacity = 0
#         self.btn2.disabled = True
#         self.add_widget(self.btn2)
#
#         # Button 3 (initially invisible)
#         self.btn3 = Button(text='Button 3')
#         self.btn3.opacity = 0
#         self.btn3.disabled = True
#         self.add_widget(self.btn3)
#
#     def check_condition(self, instance):
#         # Example condition: check if the button text is 'Button 1'
#         if instance.text == 'Button 1':
#             self.condition_met = True
#         else:
#             self.condition_met = False
#
#         self.update_visibility()
#
#     def update_visibility(self):
#         if self.condition_met:
#             self.btn2.opacity = 1
#             self.btn2.disabled = False
#             self.btn3.opacity = 1
#             self.btn3.disabled = False
#         else:
#             self.btn2.opacity = 0
#             self.btn2.disabled = True
#             self.btn3.opacity = 0
#             self.btn3.disabled = True
#
#
# class MyApp(App):
#     def build(self):
#         return MyWidget()
#
#
# if __name__ == '__main__':
#     MyApp().run()
# from kivy.app import App
# from kivy.uix.boxlayout import BoxLayout
# from kivy.uix.button import Button
# from kivy.properties import BooleanProperty
#
# def detect():
#     # Replace this with your actual object detection logic
#     print("Detecting objects...")
#
# def set_regions():
#     # Replace this with your actual region setting logic
#     print("Setting regions...")
# class MyWidget(BoxLayout):
#     condition_met = BooleanProperty(False)
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.orientation = 'vertical'
#
#         # Button 1
#         self.btn1 = Button(text='Button 1', on_press=self.check_condition)
#         self.add_widget(self.btn1)
#
#         # Button 2 (initially invisible)
#         self.btn2 = Button(text='Button 2', on_press=self.set_regions_and_detect)
#         self.btn2.opacity = 0
#         self.btn2.disabled = True
#         self.add_widget(self.btn2)
#
#         # Button 3 (initially invisible)
#         self.btn3 = Button(text='Button 3',on_press=self.detect)
#         self.btn3.opacity = 0
#         self.btn3.disabled = True
#         self.add_widget(self.btn3)
#
#         self.btn4 = Button(text='Button 4', on_press=self.check_condition)
#         self.btn4.opacity = 0
#         self.btn4.disabled = True
#         self.add_widget(self.btn4)
#
#     def check_condition(self, instance):
#         # Example condition: check if the button text is 'Button 1'
#         # This is where you can implement a more complex condition if needed
#         self.set_regions_and_detect()
#         if instance.text == 'Button 1':
#             self.condition_met = True
#         else:
#             self.condition_met = False
#
#         self.update_visibility()
#
#     def update_visibility(self):
#         # Update visibility of buttons 2 and 3 based on the condition
#         for button in [self.btn2, self.btn3, self.btn4]:
#             button.opacity = 1 if self.condition_met else 0
#             button.disabled = not self.condition_met
#
#     def set_regions_and_detect(self,instance):
#         # Call set_regions and then detect after user selects "Yes"
#         set_regions()
#         detect()
#
#     def detect(self):
#         # Call set_regions and then detect after user selects "Yes"
#         detect()
#
#
# class MyApp(App):
#     def build(self):
#         return MyWidget()
#
#
# if __name__ == '__main__':
#     MyApp().run()


from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.properties import BooleanProperty, StringProperty
import os
import tkinter as tk
import tkinter.filedialog

def select_video_file():
    root = tk.Tk()
    root.withdraw()
    file_path = tk.filedialog.askopenfilename(filetypes=[('Video files', '*.mp4 *.avi *.mkv')])
    root.destroy()
    return file_path
class MyWidget(BoxLayout):
    condition_met = BooleanProperty(False)
    video_path = StringProperty('')
    region_file_path = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # Button 1
        self.btn1 = Button(text='Select Video', on_press=self.select_video)
        self.add_widget(self.btn1)

        # Button 2 (initially invisible)
        self.btn2 = Button(text='Set Region', on_press=self.set_region)
        self.btn2.opacity = 0
        self.btn2.disabled = True
        self.add_widget(self.btn2)

        # Button 3 (initially invisible)
        self.btn3 = Button(text='Detect', on_press=self.detect)
        self.btn3.opacity = 0
        self.btn3.disabled = True
        self.add_widget(self.btn3)

        # Button 4 (initially invisible)
        self.btn4 = Button(text='Reset', on_press=self.reset_process)
        self.btn4.opacity = 0
        self.btn4.disabled = True
        self.add_widget(self.btn4)

    def select_video(self, instance):
        # Create a file chooser to select a video file
        root = tk.Tk()
        root.withdraw()
        file_path = tk.filedialog.askopenfilename(filetypes=[('Video files', '*.mp4 *.avi *.mkv')])
        root.destroy()
        print(file_path)
        # file_path = select_video_file()
        # content = BoxLayout(orientation='vertical')
        # filechooser = FileChooserListView()
        # content.add_widget(file_path)
        self.video_path = file_path

        region_file = self.video_path.rsplit('.', 1)[0] + '.p'
        if not os.path.exists(region_file):
            self.set_region_file_path()
        else:
            self.region_file_path = region_file
        print(region_file)
        print(self.video_path)
        self.condition_met = True
        self.update_visibility()
        def on_select(instance):
            self.video_path = file_path

            region_file = self.video_path.rsplit('.', 1)[0] + '.p'
            if not os.path.exists(region_file):
                self.set_region_file_path()
            else:
                self.region_file_path = region_file
            print(region_file)
            print(self.video_path)
            self.condition_met = True
            self.update_visibility()
            # popup.dismiss()

        # select_button = Button(text='Select', on_press=on_select)
        # content.add_widget(select_button)
        # popup = Popup(title='Select Video File', content=content, size_hint=(0.9, 0.9))
        # popup.open()

    def set_region_file_path(self):
        # Example function for setting the region file path
        self.region_file_path = self.video_path.rsplit('.', 1)[0] + '.p'
        # Implement the logic to set regions here
        print(f"Setting region for video: {self.video_path}")
        print(f"Region file path: {self.region_file_path}")

    def set_region(self, instance):
        # Implement the logic to set regions here
        print(f"Setting region with file path: {self.region_file_path}")
        self.detect(self.video_path)

    def detect(self, instance):
        # Implement the logic for detection here
        print(f"Detecting in video: {self.video_path}")

    def reset_process(self, instance):
        self.condition_met = False
        self.video_path = ''
        self.region_file_path = ''
        self.update_visibility()

    def update_visibility(self):
        for button in [self.btn2, self.btn3, self.btn4]:
            button.opacity = 1 if self.condition_met else 0
            button.disabled = not self.condition_met


class MyApp(App):
    def build(self):
        return MyWidget()


if __name__ == '__main__':
    MyApp().run()
