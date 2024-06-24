import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import heapq
import json


class Node:
    def __init__(self, position, g=0, h=0):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    return np.sum(np.abs(np.array(a) - np.array(b)))


def get_neighbors(current, grid):
    x, y, z = current.position
    neighbors = []
    for dx, dy, dz in [(0, 1, 0), (1, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]:
            if grid[nx, ny, nz] != 'wall':
                neighbors.append(Node((nx, ny, nz)))
    return neighbors


def a_star(grid, start, end):
    start_node = Node(start)
    end_node = Node(end)

    open_list = []
    closed_set = set()

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node.position == end_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        for neighbor in get_neighbors(current_node, grid):
            if neighbor.position in closed_set:
                continue

            neighbor.g = current_node.g + 1
            if grid[neighbor.position] == 'door':
                neighbor.g += 5  # Higher cost for doors
            elif grid[neighbor.position] == 'stair':
                neighbor.g += 2  # Moderate cost for stairs

            neighbor.h = heuristic(neighbor.position, end_node.position)
            neighbor.f = neighbor.g + neighbor.h
            neighbor.parent = current_node

            if neighbor not in open_list:
                heapq.heappush(open_list, neighbor)
            else:
                idx = open_list.index(neighbor)
                if open_list[idx].g > neighbor.g:
                    open_list[idx] = neighbor
                    heapq.heapify(open_list)

    return None  # No path found


class BIMPathfinder:
    def __init__(self, master):
        self.master = master
        self.grid = None
        self.bbox = None
        self.start = None
        self.end = None
        self.path = None

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.controls_frame = ttk.Frame(self.master)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.load_button = ttk.Button(self.controls_frame, text="Load Grid", command=self.load_grid)
        self.load_button.grid(row=0, column=0, columnspan=2)

        ttk.Label(self.controls_frame, text="Start:").grid(row=1, column=0)
        self.start_entry = ttk.Entry(self.controls_frame)
        self.start_entry.grid(row=1, column=1)

        ttk.Label(self.controls_frame, text="End:").grid(row=2, column=0)
        self.end_entry = ttk.Entry(self.controls_frame)
        self.end_entry.grid(row=2, column=1)

        self.find_path_button = ttk.Button(self.controls_frame, text="Find Path", command=self.find_path)
        self.find_path_button.grid(row=3, column=0, columnspan=2)

    def load_grid(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.grid = np.array(data['grid'])
            self.bbox = data['bbox']
            self.visualize_grid()

    def visualize_grid(self):
        if self.grid is None:
            return

        self.ax.clear()
        colors = {
            'wall': 'gray',
            'door': 'brown',
            'stair': 'red',
            'slab': 'blue',
            'other': 'green'
        }

        for element_type, color in colors.items():
            x, y, z = np.where(self.grid == element_type)
            self.ax.scatter(x, y, z, c=color, marker='s', alpha=0.1, label=element_type)

        if self.start:
            self.ax.scatter(*self.start, c='yellow', s=100, label='Start')
        if self.end:
            self.ax.scatter(*self.end, c='purple', s=100, label='End')
        if self.path:
            path_array = np.array(self.path)
            self.ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], c='cyan', linewidth=2, label='Path')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()

        self.canvas.draw()

    def find_path(self):
        if self.grid is None:
            print("Please load a grid first")
            return

        try:
            start = tuple(map(int, self.start_entry.get().split(',')))
            end = tuple(map(int, self.end_entry.get().split(',')))

            if len(start) != 3 or len(end) != 3:
                raise ValueError("Start and End must be 3D coordinates")

            self.start = start
            self.end = end

            self.path = a_star(self.grid, start, end)
            if self.path:
                print("Path found:", self.path)
            else:
                print("No path found")

            self.visualize_grid()
        except ValueError as e:
            print(f"Error: {e}")


def main():
    root = tk.Tk()
    root.title("BIM Pathfinder")
    app = BIMPathfinder(root)
    root.mainloop()

if __name__ == "__main__":
    main()