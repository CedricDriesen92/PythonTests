import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import heapq
from matplotlib.widgets import Button, Slider, TextBox, RadioButtons
import tkinter as tk
from tkinter.filedialog import askopenfilename
tk.Tk().withdraw() # part of the import if you are not using other tkinter functions
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import line_profiler_pycharm
from line_profiler_pycharm import profile
from timeit import default_timer as timer


class Node:
    def __init__(self, position, g=0, h=0):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f


class InteractiveBIMPathfinder:
    def __init__(self, filename):
        self.grids, self.bbox, self.floors, self.grid_size = self.load_grid_data(filename)
        self.start = None
        self.goal = None
        self.current_floor = 0
        self.speed = 1  # Default speed
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.canvas = self.fig.canvas
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.fig.canvas.get_tk_widget().master)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.path = None
        self.minimize_cost = True
        self.algorithm = 'A*'
        self.animated = True
        self.fps = 1
        self.setup_plot()

    def load_grid_data(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        grids = [np.array(grid) for grid in data['grids']]
        return grids, data['bbox'], data['floors'], data['grid_size']

    def setup_plot(self):
        plt.subplots_adjust(bottom=0.3)
        self.ax_prev = plt.axes([0.2, 0.15, 0.1, 0.075])
        self.ax_next = plt.axes([0.31, 0.15, 0.1, 0.075])
        self.ax_start = plt.axes([0.42, 0.15, 0.1, 0.075])
        self.ax_goal = plt.axes([0.53, 0.15, 0.1, 0.075])
        self.ax_run = plt.axes([0.64, 0.15, 0.1, 0.075])
        self.ax_speed = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.ax_pathlength = plt.axes([0.9, 0.15, 0.075, 0.03])
        self.ax.set_xlim(0, self.grids[0].shape[1])
        self.ax.set_ylim(0, self.grids[0].shape[0])
        self.ax_minimize = plt.axes([0.75, 0.25, 0.2, 0.1])
        self.ax_animate = plt.axes([0.75, 0.7, 0.2, 0.1])
        self.ax_fps = plt.axes([0.75, 0.65, 0.2, 0.03])


        self.b_prev = Button(self.ax_prev, 'Previous')
        self.b_next = Button(self.ax_next, 'Next')
        self.b_start = Button(self.ax_start, 'Set Start')
        self.b_goal = Button(self.ax_goal, 'Set Goal')
        self.b_run = Button(self.ax_run, 'Run A*')
        self.s_speed = Slider(self.ax_speed, 'Delay', 1, 100, valinit=self.speed, valstep=0.1)
        self.t_pathlength = TextBox(self.ax_pathlength, 'Path Length:', initial='0')
        self.radio_minimize = RadioButtons(self.ax_minimize, ('Cost', 'Distance'))
        self.radio_animate = RadioButtons(self.ax_animate, ('Yes', 'No'))
        self.s_fps = Slider(self.ax_fps, 'fps', 0.1, 120, valinit=self.speed, valstep=0.1)

        self.b_prev.on_clicked(self.prev_floor)
        self.b_next.on_clicked(self.next_floor)
        self.b_start.on_clicked(self.set_start_mode)
        self.b_goal.on_clicked(self.set_goal_mode)
        self.b_run.on_clicked(self.run_astar)
        self.s_speed.on_changed(self.update_speed)
        self.radio_minimize.on_clicked(self.set_minimize)
        self.radio_animate.on_clicked(self.set_animate)
        self.s_fps.on_changed(self.update_fps)

        self.mode = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.update_plot()

    def set_algorithm(self, label):
        self.algorithm = label

    def set_minimize(self, label):
        self.minimize_cost = (label == 'Cost')

    def set_animate(self, label):
        self.animated = (label == 'Yes')

    def update_speed(self, val):
        self.speed = int(val)

    def update_fps(self, val):
        self.fps = float(val)

    def grid_to_numeric(self, grid):
        element_types = ['empty', 'wall', 'door', 'stair', 'floor']
        numeric_grid = np.zeros_like(grid, dtype=int)
        for i, element_type in enumerate(element_types):
            numeric_grid[grid == element_type] = i
        return numeric_grid

    def update_plot(self):
        self.ax.clear()
        colors = ['white', 'black', 'orange', 'red', 'lavenderblush']
        color_map = ListedColormap(colors)

        grid = self.grid_to_numeric(self.grids[self.current_floor])
        self.ax.imshow(grid.T, cmap=color_map, interpolation='nearest')

        if self.start and self.start[2] == self.current_floor:
            self.ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        if self.goal and self.goal[2] == self.current_floor:
            self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)

        if self.path:
            self.visualize_path()
        self.ax.set_xlim(self.ax.get_xlim())  # Preserve zoom level
        self.ax.set_ylim(self.ax.get_ylim())  # Preserve zoom level
        self.ax.set_title(f'Floor {self.current_floor + 1}')
        self.ax.axis('off')
        plt.draw()

    def prev_floor(self, event):
        if self.current_floor > 0:
            self.current_floor -= 1
            self.update_plot()

    def next_floor(self, event):
        if self.current_floor < len(self.grids) - 1:
            self.current_floor += 1
            self.update_plot()

    def set_start_mode(self, event):
        self.mode = 'start'

    def set_goal_mode(self, event):
        self.mode = 'goal'

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        if self.mode == 'start':
            self.start = (x, y, self.current_floor)
            print(f"Start set to: {self.start}")
        elif self.mode == 'goal':
            self.goal = (x, y, self.current_floor)
            print(f"Goal set to: {self.goal}")
        self.mode = None
        self.update_plot()

    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) * self.grid_size

    def get_neighbors(self, current):
        x, y, z = current.position
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grids[z].shape[0] and 0 <= ny < self.grids[z].shape[1]:
                if self.grids[z][nx, ny] != 'wall':
                    neighbors.append(Node((nx, ny, z)))

        if self.grids[z][x, y] == 'stair':
            for nz in range(len(self.grids)):
                if nz != z and self.grids[nz][x, y] == 'stair':
                    neighbors.append(Node((x, y, nz)))

        return neighbors

    def run_algorithm(self, event):
        if not self.start or not self.goal:
            print("Please set both start and goal points.")
            return
        if self.algorithm == 'A*':
            self.run_astar()
        elif self.algorithm == 'Dijkstra':
            self.run_dijkstra()
        elif self.algorithm == 'BFS':
            self.run_bfs()

    @profile
    def run_astar(self, event):
        if not self.start or not self.goal:
            print("Please set both start and goal points.")
            return
        fps = self.fps
        time0 = timer()
        self.path = None  # Clear the previous path
        open_list = []
        closed_set = set()
        start_node = Node(self.start)
        goal_node = Node(self.goal)

        heapq.heappush(open_list, (start_node.f, start_node))

        progress_counter = 0

        progress_threshold = max(1, int(100 / self.speed))

        while open_list:
            current_node = heapq.heappop(open_list)[1]
            if current_node.position == goal_node.position:
                path = []
                path_length = 0
                while current_node:
                    path.append(current_node.position)
                    if current_node.parent:
                        path_length += self.heuristic(current_node.position, current_node.parent.position)
                    current_node = current_node.parent
                self.path = path[::-1]


                self.t_pathlength.set_val(f"{path_length:.2f}")
                self.visualize_path()
                return

            closed_set.add(current_node.position)

            for neighbor in self.get_neighbors(current_node):
                if neighbor.position in closed_set:
                    continue

                neighbor.g = current_node.g + self.grid_size
                if self.grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'door':
                    if self.minimize_cost:
                        neighbor.g += 5 * self.grid_size  # Higher cost for doors

                elif self.grids[neighbor.position[2]][neighbor.position[0], neighbor.position[1]] == 'stair':
                    if self.minimize_cost:
                        neighbor.g += 1.0 * self.grid_size  # Moderate cost for stairs

                neighbor.h = self.heuristic(neighbor.position, goal_node.position)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node

                if not any(node.position == neighbor.position for _, node in open_list):
                    heapq.heappush(open_list, (neighbor.f, neighbor))
                else:
                    # Update existing node if this path is better
                    for i, (f, node) in enumerate(open_list):
                        if node.position == neighbor.position and neighbor.g < node.g:
                            open_list[i] = (neighbor.f, neighbor)
                            heapq.heapify(open_list)
                            break

            progress_counter += 1
            if progress_threshold == 0 or progress_counter >= progress_threshold:
                if timer()-time0 > 1.0/fps:
                    time0 = timer()
                    current_path = []
                    temp_node = current_node
                    current_path_length = 0
                    while temp_node:
                        current_path.append(temp_node.position)
                        if temp_node.parent:
                            current_path_length += self.heuristic(temp_node.position, temp_node.parent.position)
                        temp_node = temp_node.parent
                    current_path = current_path[::-1]
                    self.t_pathlength.set_val(f"{current_path_length:.2f}")
                    self.visualize_progress(closed_set, [node for _, node in open_list], current_path)
                    progress_counter = 0

        print("No path found.")

    @profile
    def visualize_progress(self, closed_set, open_list, current_path):
        self.ax.clear()
        colors = ['white', 'black', 'orange', 'red', 'lavenderblush']
        color_map = ListedColormap(colors)

        grid = self.grid_to_numeric(self.grids[self.current_floor])
        self.ax.imshow(grid.T, cmap=color_map, interpolation='nearest')

        # Plot closed set
        closed_on_floor = [node for node in closed_set if node[2] == self.current_floor]
        if closed_on_floor:
            closed_x, closed_y = zip(*[(node[0], node[1]) for node in closed_on_floor])
            self.ax.scatter(closed_x, closed_y, color='blue', alpha=0.1, s=20)

        # Plot open list
        open_on_floor = [node.position for node in open_list if node.position[2] == self.current_floor]
        if open_on_floor:
            open_x, open_y = zip(*[(node[0], node[1]) for node in open_on_floor])
            self.ax.scatter(open_x, open_y, color='aqua', alpha=0.4, s=20)

        # Plot current path
        current_path_on_floor = [node for node in current_path if node[2] == self.current_floor]
        if len(current_path_on_floor) > 1:
            path_x, path_y = zip(*[(node[0], node[1]) for node in current_path_on_floor])
            self.ax.plot(path_x, path_y, color='yellow', linewidth=2)

        # Plot start and goal
        if self.start and self.start[2] == self.current_floor:
            self.ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        if self.goal and self.goal[2] == self.current_floor:
            self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)

        self.ax.set_title(f'Floor {self.current_floor + 1}')
        self.ax.axis('off')
        plt.draw()
        plt.pause(0.01)

    def visualize_path(self):
        self.ax.clear()
        colors = ['white', 'gray', 'brown', 'red', 'beige']
        color_map = ListedColormap(colors)

        grid = self.grid_to_numeric(self.grids[self.current_floor])
        self.ax.imshow(grid.T, cmap=color_map, interpolation='nearest')

        if self.path:
            path_on_floor = [node for node in self.path if node[2] == self.current_floor]
            for i in range(len(path_on_floor) - 1):
                start = path_on_floor[i]
                end = path_on_floor[i + 1]

                # Check if this segment is continuous or a jump
                if (abs(end[0] - start[0]) > 1 or abs(end[1] - start[1]) > 1):
                    # Discontinuous segment - draw in green
                    self.ax.plot([start[0], end[0]], [start[1], end[1]], color='green', linewidth=2, linestyle='--')
                else:
                    # Continuous segment - draw in blue
                    self.ax.plot([start[0], end[0]], [start[1], end[1]], color='blue', linewidth=2)

        if self.start and self.start[2] == self.current_floor:
            self.ax.plot(self.start[0], self.start[1], 'go', markersize=10)
        if self.goal and self.goal[2] == self.current_floor:
            self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10)

        self.ax.set_title(f'Floor {self.current_floor + 1}')
        self.ax.axis('off')
        plt.draw()
        plt.pause(0.001)


def main():
    fn = askopenfilename(filetypes=[("json files", "*.json")])
    pathfinder = InteractiveBIMPathfinder(fn)
    plt.show()


if __name__ == "__main__":
    main()