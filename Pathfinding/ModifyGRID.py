import numpy as np
import matplotlib.pyplot as plt
import json
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from matplotlib.widgets import Button, RadioButtons

class InteractiveGridEditor:
    def __init__(self, grids, grid_size, floors, bbox):
        self.grids = grids
        self.grid_size = grid_size
        self.floors = floors
        self.bbox = bbox
        self.current_floor = 0
        self.current_element = 'wall'
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.setup_plot()

    def setup_plot(self):
        plt.subplots_adjust(bottom=0.2, right=0.8)
        self.ax_prev = plt.axes([0.1, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.21, 0.05, 0.1, 0.075])
        self.ax_save = plt.axes([0.32, 0.05, 0.1, 0.075])
        self.ax_radio = plt.axes([0.81, 0.3, 0.15, 0.3])

        self.b_prev = Button(self.ax_prev, 'Previous Floor')
        self.b_next = Button(self.ax_next, 'Next Floor')
        self.b_save = Button(self.ax_save, 'Save')
        self.radio = RadioButtons(self.ax_radio, ('wall', 'floor', 'door', 'stair', 'empty'))

        self.b_prev.on_clicked(self.prev_floor)
        self.b_next.on_clicked(self.next_floor)
        self.b_save.on_clicked(self.save_grids)
        self.radio.on_clicked(self.set_element)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        colors = {
            'wall': 'black',
            'floor': 'lavenderblush',
            'door': 'orange',
            'stair': 'red',
            'empty': 'lightgray'
        }

        for element_type in ['empty', 'wall', 'stair', 'floor', 'door']:
            y, x = np.where(self.grids[self.current_floor] == element_type)
            self.ax.scatter(x, y, c=colors[element_type], marker='s', s=self.grid_size * 80, edgecolors='none')

        self.ax.set_title(f'Floor {self.current_floor + 1} (Elevation: {self.floors[self.current_floor]["elevation"]:.2f}m)')
        self.ax.set_aspect('equal', 'box')
        self.ax.invert_yaxis()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.draw()

    def prev_floor(self, event):
        if self.current_floor > 0:
            self.current_floor -= 1
            self.update_plot()

    def next_floor(self, event):
        if self.current_floor < len(self.grids) - 1:
            self.current_floor += 1
            self.update_plot()

    def set_element(self, label):
        self.current_element = label

    def on_click(self, event):
        if event.inaxes == self.ax:
            self.paint(event)

    def on_motion(self, event):
        if event.inaxes == self.ax and event.button == 1:  # Left mouse button
            self.paint(event)

    def paint(self, event):
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < self.grids[self.current_floor].shape[1] and 0 <= y < self.grids[self.current_floor].shape[0]:
            self.grids[self.current_floor][y, x] = self.current_element
            self.update_plot()

    def save_grids(self, event):
        filename = asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filename:
            data = {
                'grids': [grid.tolist() for grid in self.grids],
                'bbox': self.bbox,
                'grid_size': self.grid_size,
                'floors': self.floors
            }
            with open(filename, 'w') as f:
                json.dump(data, f)
            print(f"Grids saved to {filename}")

def load_grid_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    grids = [np.array(grid) for grid in data['grids']]
    return grids, data['grid_size'], data['floors'], data['bbox']

def main():
    tk.Tk().withdraw()
    fn = askopenfilename(filetypes=[("JSON files", "*.json")])
    if fn:
        grids, grid_size, floors, bbox = load_grid_data(fn)
        editor = InteractiveGridEditor(grids, grid_size, floors, bbox)
        plt.show()
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()