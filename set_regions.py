import os
import numpy as np
import cv2
import pickle
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector
from matplotlib.collections import PatchCollection

class SmartCarParking:
    def __init__(self, video_path, save_path):
        self.video_path = video_path
        self.save_path = save_path if save_path.endswith(".p") else save_path + ".p"
        self.points = []
        self.prev_points = []
        self.patches = []
        self.total_points = []
        self.breaker = False
        self.rgb_image = None
        self.globSelect = None
        self.load_existing_data()

    def load_existing_data(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    self.total_points = pickle.load(f)
                self.patches = [Polygon(p) for p in self.total_points]
            except (EOFError, pickle.UnpicklingError):
                print("Error loading data from", self.save_path, ". Starting fresh.")
                self.total_points = []
                self.patches = []

    def save_data(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.total_points, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Data saved in", self.save_path + " file")

    def break_loop(self, event):
        if event.key == 'b':
            self.globSelect.disconnect()
            self.save_data()
            exit()

    def onkeypress(self, event):
        if event.key == 'n':
            pts = np.array(self.points, dtype=np.int32)
            if self.points != self.prev_points and len(set(tuple(p) for p in self.points)) == 4:
                print("Points:", pts)
                self.patches.append(Polygon(pts))
                self.total_points.append(pts)
                self.prev_points = self.points
        elif event.key == 'q':
            plt.close()  # Close the current figure to start a new quadrilateral

    def initialize_video_capture(self):
        video_capture = cv2.VideoCapture(self.video_path)
        success, frame = video_capture.read()
        if success:
            self.rgb_image = frame[:, :, ::-1]
        video_capture.release()

    def run(self):
        self.initialize_video_capture()
        while True:
            fig, ax = plt.subplots()
            ax.imshow(self.rgb_image)

            p = PatchCollection(self.patches, alpha=0.7)
            p.set_array(10 * np.ones(len(self.patches)))
            ax.add_collection(p)

            self.globSelect = SelectFromCollection(ax, self)
            plt.connect('key_press_event', self.onkeypress)
            plt.connect('key_press_event', self.break_loop)

            paragraph = """
            Select a region in the figure by enclosing them within a quadrilateral:
                - Press 'f' to toggle fullscreen.
                - Press 'esc' to discard the current quadrilateral.
                - Hold 'mouse left button' to move a single vertex.
                - After marking a quadrilateral:
                - Press 'n' to save the current quadrilateral.
                - Press 'q' to start marking a new quadrilateral.
                - When finished, press 'b' to exit the program.
            """
            plt.figtext(0.1, 0, paragraph, wrap=True, horizontalalignment='left', fontsize=12)
            fig.subplots_adjust(bottom=0.3)
            plt.show()

            self.globSelect.disconnect()

            if self.breaker:
                break

class SelectFromCollection:
    def __init__(self, ax, parent):
        self.canvas = ax.figure.canvas
        self.parent = parent
        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []

    def onselect(self, verts):
        self.parent.points = verts
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.canvas.draw_idle()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Path of video file")
    parser.add_argument('--out_file', help="Name of the output file")
    args = parser.parse_args()

    app = SmartCarParking(args.video_path, args.out_file)
    app.run()
